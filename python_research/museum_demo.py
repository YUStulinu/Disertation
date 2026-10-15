"""
Module 2 - Week 6 Day 3: museum object placement demo.

The unified pipeline (pipeline.py) with settings for the professor's use case,
museum artefacts placed into new halls: YOLOv8m + SAM 2.1 segmentation, LaMa
inpainting, matted cut-out, depth-based scale (--height, in metres), ensemble
harmonization, camera match, a faint floor reflection (polished museum floors)
and a contact shadow - plus a figure per background:
Source | Inpainted | Composed (raw) | Final | Final, enlarged around the object.

Saves <src>_on_<bg>_final.jpg with its placed mask <src>_on_<bg>_mask.png
(add_shadow.py --batch data/museum_demo can redo the shadow with other settings).

Usage:
    python museum_demo.py --source amphora.jpg --background hall1.jpg hall2.jpg --height 0.51 --pos 0.5 0.86
"""

from common.imaging import crop_object, labeled_grid, show, write_image
from common.paths import DATA_DIR
from pipeline import build_parser, run

OUTPUT_DIR = DATA_DIR / "museum_demo"


def main() -> None:
    parser = build_parser("Museum object placement demo - Week 6 Day 3", cls="vase", yolo="m", inpaint="lama",
                          scale=0.35, pos=(0.65, 0.82), reflection=0.2, out_dir=OUTPUT_DIR)
    args = parser.parse_args()
    show_result, args.no_show = not args.no_show, True
    for out in run(args):
        figure = labeled_grid([(out["source"], "1 - Source"), (out["inpainted"], f"2 - Inpainted ({args.inpaint})"),
                               (out["composed"], "3 - Composed (raw)"), (out["final"], "4 - Final"),
                               (crop_object(out["final"], out["alpha"]), "5 - Final (detail)")])
        print("    saved", write_image(args.out_dir / f"{out['stem']}_demo.jpg", figure).name)
        if show_result:
            show(f"Museum demo - {out['stem']}", figure)


if __name__ == "__main__":
    main()
