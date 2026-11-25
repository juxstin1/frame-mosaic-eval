"""Fixed question templates. Every prompt = layout description + task question + strict answer format."""
from gen.common import SHAPES
from mosaic.builder import GRIDS, Condition


def layout(cond: Condition) -> str:
    n = cond.n
    if cond.fmt == "separate":
        s = f"These {n} images are consecutive frames from a video, in order: frame 1 is the first image, frame {n} the last."
    else:
        rows, cols = GRIDS[n]
        order = ("Read the grid left to right, top to bottom" if cond.order == "row" else
                 "Read the grid in snake order: the first row left to right, the second row right to left, "
                 "and so on, alternating")
        s = (f"This image is a grid of {n} consecutive frames from a video ({rows} rows x {cols} columns). "
             f"{order}: frame 1 is the first cell, frame {n} the last.")
    if cond.timestamps:
        s += {"frame": " Each frame is labelled with its frame number in its top-left corner.",
              "col": " Each frame is labelled in its top-left corner with its column number (c1 = leftmost column).",
              "row": " Each frame is labelled in its top-left corner with its row number (r1 = top row)."
              }[cond.label_mode]
    return s


QUESTIONS = {
    "moving_dot": ("A dot moves across the frames. At some frame it changes colour from blue to red and stays red. "
                   "What is the number of the first frame in which the dot is red? Answer with the frame number only."),
    "counter": ("Each frame shows a two-digit number. List the number shown in every frame, in frame order, "
                "separated by commas. Answer with the list only."),
    "event_order": ("Several shapes each appear in exactly one frame. List the shapes in the order they appear, "
                    "separated by commas, using only these names: " + ", ".join(SHAPES) +
                    ". Answer with the list only."),
    "count_events": "In how many frames does a yellow star appear? Answer with the number only.",
    "tile_probe": ("Exactly one frame contains a red star. What is the number of that frame? "
                   "Answer with the frame number only."),
}


REASONING_FORMAT = "You may work step by step. End with a final line of the form \"Answer: <number>\"."


def prompt(task: str, cond: Condition, reasoning: bool = False) -> str:
    """reasoning=True swaps the strict "Answer with ... only." sentence for REASONING_FORMAT."""
    q = QUESTIONS[task]
    if reasoning:
        q = q[:q.index(" Answer with")] + " " + REASONING_FORMAT
    return layout(cond) + "\n\n" + q


def max_answer_tokens(task: str, n: int) -> int:
    return 16 + 4 * n if task in ("counter", "event_order") else 16
