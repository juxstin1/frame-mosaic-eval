"""Matched 768px hosted comparison. Smoke sequences are excluded from accuracy."""
import argparse, base64, hashlib, io, json, re, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image
from gen.common import FRAME_SIZE, rng_for
from gen.tasks import TASKS
from mosaic.builder import Condition, build
from eval.questions import prompt
from eval.score import score, apply_truncation_rule
MODELS={'mistral':'mistralai/Mistral-Small-3.2-24B-Instruct-2506','scout':'meta-llama/Llama-4-Scout-17B-16E-Instruct'}
SYSTEM='Respond with a single integer only. Do not explain or show working.'
PROTOCOL='matched768_integer_system_v1'
FORMAT=None

def make_case(task,n,labels,seq):
    frames,answer,_=TASKS[task](n,rng_for(task,n,seq,0),FRAME_SIZE)
    cond=Condition(n=n,timestamps=labels,align=48)
    images,meta=build(frames,cond)
    im=images[0].resize((768,768),Image.Resampling.LANCZOS)
    buf=io.BytesIO(); im.save(buf,format='PNG'); png=buf.getvalue()
    return png,answer,meta,prompt(task,cond)

def run_model(alias,key,folder,phases=('smoke','sweep'),endpoint='https://api.deepinfra.com/v1/openai/chat/completions',extra_body=None):
    model=MODELS[alias]
    for phase in phases:
        ns=[4,64] if phase=='smoke' else [4,9,16,36,64]
        seqs=[99] if phase=='smoke' else range(5)
        total=3*len(ns)*2*len(seqs); done=0; gate=True
        with (folder/(alias+'_'+phase+'.jsonl')).open('x',encoding='utf-8') as fh:
            for task in ['moving_dot','tile_probe','count_events']:
                for n in ns:
                    for labels in [False,True]:
                        for seq in seqs:
                            png,answer,meta,q=make_case(task,n,labels,seq)
                            messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}},{'type':'text','text':q}]}]
                            body={'model':model,'messages':messages,'temperature':0,'max_tokens':32,'stream':False}
                            if FORMAT is not None: body['response_format']=FORMAT
                            if extra_body: body.update(extra_body)
                            rec={'protocol':PROTOCOL,'response_format':FORMAT,'model':model,'phase':phase,'task':task,'n':n,'labels':labels,'seq_index':seq,'base_seed':0,'condition':meta['condition'],'source_image_wh':meta['image_wh'],'image_wh':[[768,768]],'image_sha256':hashlib.sha256(png).hexdigest(),'system_prompt':SYSTEM,'prompt':q,'temperature':0,'max_tokens':32,'time_utc':datetime.now(timezone.utc).isoformat()}
                            req=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                            start=time.perf_counter()
                            try:
                                with urllib.request.urlopen(req,timeout=90) as r: response=json.load(r)
                                rec['response']=response; rec['latency_s']=round(time.perf_counter()-start,3)
                                choice=response['choices'][0]; raw=choice['message'].get('content') or ''
                                rec.update(raw_answer=raw,finish_reason=choice.get('finish_reason'),**score(task,answer,raw))
                                apply_truncation_rule(rec)
                                rec['single_integer']=bool(re.fullmatch(r'\s*\d+[.]?\s*',raw))
                                if response.get('model')!=model: raise ValueError('model identity mismatch')
                                if not rec['single_integer'] or rec['finish_reason']!='stop': gate=False
                            except Exception as e:
                                rec['error']={'type':type(e).__name__,'http_status':getattr(e,'code',None)}
                                fh.write(json.dumps(rec)+'\n'); fh.flush()
                                print(json.dumps({'model':alias,'phase':phase,'error':rec['error']}),flush=True)
                                return False
                            fh.write(json.dumps(rec)+'\n'); fh.flush(); done+=1
                            if phase=='smoke' or done%10==0:
                                print(json.dumps({'model':alias,'phase':phase,'done':done,'total':total,'answer':raw,'finish_reason':rec['finish_reason']}),flush=True)
        if phase=='smoke':
            print(alias+' SMOKE_GATE '+str(gate),flush=True)
            if not gate: return False
    print(alias+' SWEEP_COMPLETE',flush=True)
    return True

def main():
    global FORMAT, PROTOCOL
    p=argparse.ArgumentParser(); p.add_argument('--key-file',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--integer-schema',action='store_true')
    a=p.parse_args()
    if a.integer_schema:
        FORMAT={'type':'json_schema','json_schema':{'name':'integer_answer','strict':True,'schema':{'type':'integer'}}}
        PROTOCOL='matched768_integer_schema_v1'
    key=next(s.split('=',1)[1].strip().strip('\"\'') for s in a.key_file.read_text().splitlines() if s.strip().startswith('DEEPINFRA_API_KEY='))
    assert key
    a.out.mkdir(parents=True,exist_ok=False)
    manifest={'protocol':PROTOCOL,'response_format':FORMAT,'models':MODELS,'system_prompt':SYSTEM,'smoke_per_model':12,'sweep_per_model':150,'seed':0,'tasks':['moving_dot','tile_probe','count_events'],'n':[4,9,16,36,64],'seqs':[0,1,2,3,4],'labels':[False,True],'output_cap':32,'temperature':0,'input_size':[768,768],'note':'Both models get identical PNG hashes and prompts. Native preprocessing remains model-specific. Smoke sequence 99 excluded from sweep. Numeric format gate allows one trailing period; it does not gate on accuracy.'}
    manifest['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.out/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('RUN_STARTED '+a.out.name,flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_model,alias,key,a.out) for alias in MODELS]
        results=[f.result() for f in jobs]
    print('RUN_FINISHED '+str(dict(zip(MODELS,results))),flush=True)
    if not all(results): raise SystemExit(1)
if __name__=='__main__': main()