"""Build the shareable report page: docs/report_page_template.html + figures/*.png -> build/frame_mosaic_capacity.html.

python -m analysis.build_page        (then publish build/frame_mosaic_capacity.html; numbers in the template are typed
                                       in by hand from REPORT.md / analysis/*.md, so update them there first)
"""
import base64
from pathlib import Path

FIGS = {"FIG_N": "accuracy_vs_n", "FIG_T": "accuracy_vs_tokens_per_frame", "FIG_H": "position_heatmap",
        "FIG_P": "position_4x4", "FIG_O": "error_offsets", "FIG_S": "position_4x4_snake"}


def main():
    s = Path("docs/report_page_template.html").read_text(encoding="utf-8")
    for key, name in FIGS.items():
        data = base64.b64encode(Path(f"figures/{name}.png").read_bytes()).decode()
        s = s.replace("{{" + key + "}}", "data:image/png;base64," + data)
    assert "{{" not in s, "unfilled placeholder"
    out = Path("build/frame_mosaic_capacity.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(s, encoding="utf-8")
    print(out, f"{len(s) // 1024} KB")


if __name__ == "__main__":
    main()
