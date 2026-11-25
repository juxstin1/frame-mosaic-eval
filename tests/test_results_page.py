"""Offline checks for reasoning labels and strict score verification."""
import unittest

from analysis.results_page import check_score, token_stats


def row(tokens=2, reasoning=None, claude=False):
    usage = {"output_tokens" if claude else "completion_tokens": tokens}
    if reasoning is not None:
        usage["output_tokens_details" if claude else "completion_tokens_details"] = {
            "thinking_tokens" if claude else "reasoning_tokens": reasoning}
    return {"response": {"usage": usage}}


class ResultsPageTests(unittest.TestCase):
    def test_missing_reasoning_is_not_zero(self):
        stat = token_stats([row(), row()])
        self.assertIsNone(stat["reasoning_total"])
        self.assertEqual(stat["reasoning_reported"], 0)
        self.assertEqual(stat["thinking_note"], "thinking unconfirmed")

    def test_explicit_zero_and_length_screen_are_distinct(self):
        short = token_stats([row(6, 0), row(7, 0)])
        long = token_stats([row(32, 0), row(32, 0)])
        self.assertEqual(short["reasoning_total"], 0)
        self.assertEqual(short["thinking_note"], "provider reports zero reasoning tokens")
        self.assertEqual(long["reasoning_total"], 0)
        self.assertEqual(long["thinking_note"], "thinking unconfirmed")
        self.assertIn("not proof", long["note_reasons"][0])

    def test_positive_reasoning_with_empty_text_is_flagged(self):
        stat = token_stats([row(7, 2)])
        self.assertEqual(stat["reasoning_text_responses"], 0)
        self.assertEqual(stat["reasoning_positive_responses"], 1)
        self.assertEqual(stat["thinking_note"], "thinking unconfirmed")

    def test_claude_usage_and_partial_counter_coverage(self):
        stat = token_stats([row(9, 0, True), row(32, 32, True), row(0, None, True)])
        self.assertEqual(stat["completion_total"], 41)
        self.assertEqual(stat["completion_median"], 9)
        self.assertEqual(stat["reasoning_total"], 32)
        self.assertEqual(stat["reasoning_reported"], 2)

    def test_cutoff_correct_looking_answer_must_stay_wrong(self):
        record = {"raw_answer": '{"answer": 3}', "finish_reason": "length",
                  "truth": 3, "correct": False, "parsed": None, "single_integer": False}
        check_score(record, "object")
        record["correct"] = True
        with self.assertRaises(ValueError):
            check_score(record, "object")

    def test_extra_object_fields_cannot_score(self):
        record = {"raw_answer": '{"answer": 3, "explanation": "3"}', "finish_reason": "stop",
                  "truth": 3, "correct": False, "parsed": None, "single_integer": False}
        check_score(record, "object")


if __name__ == "__main__":
    unittest.main()
