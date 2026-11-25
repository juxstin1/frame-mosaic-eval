from eval.score import score


def test_frame_tasks():
    a = {"change_frame": 7}
    assert score("moving_dot", a, "7")["correct"]
    assert score("moving_dot", a, "Frame 8.")["within1"] and not score("moving_dot", a, "8")["correct"]
    assert score("moving_dot", a, "red")["parsed"] is None
    assert score("count_events", {"k": 3}, "3")["correct"]


def test_counter():
    r = score("counter", {"values": [1, 2, 3, 4]}, "01, 02, 9, 04")
    assert not r["correct"] and r["partial"] == 0.75
    assert score("counter", {"values": [1, 2]}, "1,2")["correct"]


def test_event_order():
    t = {"order": ["star", "circle", "square"]}
    assert score("event_order", t, "star, circle, square")["correct"]
    r = score("event_order", t, "circle, star, square")
    assert not r["correct"] and abs(r["partial"] - 2 / 3) < 1e-9


def test_truncated_answer_is_unparsed():
    from eval.score import apply_truncation_rule
    r = {"finish_reason": "length", "parsed": 1, "correct": True, "within1": True, "partial": 1.0}
    assert apply_truncation_rule(r)["parsed"] is None and not r["correct"] and r["truncated"]
    assert apply_truncation_rule({"finish_reason": "stop", "correct": True})["correct"]


def test_reasoning_mode_scores_final_answer_line():
    raw = "Frame 3 is blue, frame 4 is red.\nAnswer: 4"
    assert score("moving_dot", {"change_frame": 4}, raw, reasoning=True)["correct"]
    assert score("moving_dot", {"change_frame": 4}, "frames 1-3 blue, 4 red", reasoning=True)["parsed"] is None
    assert score("moving_dot", {"change_frame": 4}, "Answer: 2 ... wait.\n**Answer:** 4", reasoning=True)["parsed"] == 4


def test_reasoning_prompt_swaps_format():
    from eval.questions import REASONING_FORMAT, prompt
    from mosaic.builder import Condition
    p = prompt("tile_probe", Condition(n=16), reasoning=True)
    assert p.endswith(REASONING_FORMAT) and "number only" not in p
    assert prompt("tile_probe", Condition(n=16)).endswith("Answer with the frame number only.")
