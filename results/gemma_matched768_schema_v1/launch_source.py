"""Run local Gemma on the same 150 inputs and integer schema as the hosted sweep."""
import hashlib,json
from pathlib import Path
from runner import hosted_matched768 as h

def main():
    root=Path.cwd()
    server=json.loads((root/'logs/gemma_matched768_server.json').read_text())
    h.MODELS={'gemma':server['alias']}
    h.FORMAT={'type':'json_schema','json_schema':{'name':'integer_answer','strict':True,'schema':{'type':'integer'}}}
    h.PROTOCOL='matched768_integer_schema_v1'
    out=root/'results/gemma_matched768_schema_v1'
    out.mkdir(exist_ok=False)
    manifest=json.loads((root/'results/deepinfra_matched768_schema_v1/protocol.json').read_text())
    manifest.update(models=h.MODELS,server=server,smoke_per_model=0,thinking=False,runner_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),note='Same 150 PNGs, system/user prompts and integer schema as hosted comparison. No extra smoke calls. Local thinking explicitly disabled.')
    (out/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (out/'runner_source.py').write_bytes(Path(h.__file__).read_bytes())
    print('GEMMA_SWEEP_STARTED 150 cases',flush=True)
    ok=h.run_model('gemma','',out,phases=('sweep',),endpoint='http://127.0.0.1:8098/v1/chat/completions',extra_body={'chat_template_kwargs':{'enable_thinking':False}})
    if not ok: raise SystemExit(1)
if __name__=='__main__': main()