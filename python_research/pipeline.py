"""
Module 2 - Unified composition pipeline (Week 4 Day 3).

  1. Segmentation   - YOLOv8-seg finds the object; SAM 2.1 refines its mask (--no-sam: YOLO mask only)
  2. Inpainting     - fill the hole in the source image (TELEA or LaMa)
  3. Extraction     - RGBA cut-out; with matting the alpha follows the photo's edges and the edge
                      colours are cleaned of the old background (--no-matting: blurred mask edge)
  4. Composition    - metric depth of the background (Depth Anything V2) gives the floor geometry;
                      with --height the object is scaled from its real size; Real-ESRGAN
                      super-resolves the cut-out when it has to be enlarged (--no-sr to skip)
  5. Harmonization  - colour / light: ensemble of three iHarmony4 networks (default) or another method
  6. Camera match   - object sharpness and sensor noise matched to the scene (--no-camera to skip)
  7. Reflection     - mirror image in a glossy floor (--reflection 0-1, default off)
  8. Shadow         - soft contact shadow in floor perspective (default), directional cast shadow or none

Outputs per background in --out-dir: <src>_on_<bg>.jpg (raw composite), _mask.png (placed
alpha, used by harmonize.py / add_shadow.py), _harmonized.jpg, _final.jpg (+ shadow) and
_comparison.jpg (depth map | composed | harmonized | final).

Usage:
    python pipeline.py --source photo.jpg --background bg.jpg --class person
    python pipeline.py --source museum.jpg --background hall1.jpg hall2.jpg --class vase --yolo m --inpaint lama \\
        --height 0.51 --pos 0.6 0.85 --reflection 0.25
    python pipeline.py ... --no-sam --no-matting --harmony harmonizer --no-camera --shadow none   # earlier version
"""

import argparse
from pathlib import Path

import numpy as np

from common.composition import add_reflection, add_shadow, inpaint, place
from common.harmonization import METHODS, harmonize, match_camera
from common.imaging import labeled_grid, read_image, show, write_image
from common.paths import DATA_DIR
from common.scene import depth_map, depth_preview, floor_foreshortening, object_scale
from common.segmentation import extract_foreground, load_yolo, segment

OUTPUT_DIR = DATA_DIR / "pipeline_results"


def build_parser(description: str = "Module 2 unified pipeline - Week 4 Day 3", **defaults) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--source", required=True, help="Image containing the object to extract")
    p.add_argument("--background", nargs="+", required=True, dest="backgrounds", help="One or more new backgrounds")
    p.add_argument("--class", dest="cls", help="YOLO class to extract (e.g. person, vase)")
    p.add_argument("--index", type=int, default=-1, help="Pick the Nth detection instead")
    p.add_argument("--yolo", choices=["n", "m"], default="n", help="YOLOv8-seg nano or medium")
    p.add_argument("--sam", action=argparse.BooleanOptionalAction, default=True, help="Refine the mask with SAM 2.1")
    p.add_argument("--inpaint", choices=["telea", "lama"], default="telea")
    p.add_argument("--matting", action=argparse.BooleanOptionalAction, default=True, help="Soft, decontaminated edges")
    p.add_argument("--harmony", choices=METHODS, default="ensemble")
    p.add_argument("--camera", action=argparse.BooleanOptionalAction, default=True, help="Match blur and noise")
    p.add_argument("--shadow", choices=["none", "ellipse", "cast"], default="ellipse")
    p.add_argument("--scale", type=float, default=0.5, help="Object height / background height (without --height)")
    p.add_argument("--height", type=float, metavar="METRES", help="Real object height: scale it from the scene depth")
    p.add_argument("--scene", choices=["indoor", "outdoor"], default="indoor", help="Depth model for the backgrounds")
    p.add_argument("--sr", action=argparse.BooleanOptionalAction, default=True, help="Super-resolve enlarged cut-outs")
    p.add_argument("--reflection", type=float, default=0.0, metavar="0-1", help="Mirror the object in a glossy floor")
    p.add_argument("--pos", nargs=2, type=float, default=(0.5, 0.9), metavar=("CX", "CY"),
                   help="Bottom-centre of the object as 0-1 fractions of the background")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--no-show", action="store_true")
    p.set_defaults(**defaults)
    return p


def run(args) -> list[dict]:
    source, src_stem = read_image(args.source), Path(args.source).stem
    print(f"[1] Segmentation (YOLOv8{args.yolo}-seg{' + SAM 2.1' if args.sam else ''})")
    mask, _ = segment(load_yolo(args.yolo), source, args.cls, args.index, args.sam)
    print(f"[2] Inpainting the source ({args.inpaint})")
    inpainted = inpaint(source, mask, args.inpaint)
    write_image(args.out_dir / f"{src_stem}_inpainted_{args.inpaint}.jpg", inpainted)
    foreground = extract_foreground(source, mask, inpainted, args.matting)

    object_rows = np.nonzero(foreground[..., 3].max(1) > 127)[0]
    pad_ratio = foreground.shape[0] / (np.ptp(object_rows) + 1)  # the cut-out includes a small margin
    outputs = []
    for bg_path in args.backgrounds:
        stem, bg = f"{src_stem}_on_{Path(bg_path).stem}", read_image(bg_path)
        anchor = (min(round(args.pos[0] * bg.shape[1]), bg.shape[1] - 1), min(round(args.pos[1] * bg.shape[0]), bg.shape[0] - 1))
        depth = depth_map(bg, args.scene)
        scale = object_scale(depth, anchor, args.height) * pad_ratio if args.height else args.scale
        tilt = floor_foreshortening(depth, anchor)
        print(f"[3-8] {stem}: scale {scale:.3f} (depth at base {depth[anchor[1], anchor[0]]:.1f} m), floor tilt "
              f"{tilt:.2f} -> harmonize ({args.harmony}) -> camera -> reflection {args.reflection} -> shadow ({args.shadow})")
        composed, alpha = place(foreground, bg, scale, tuple(args.pos), sr=args.sr)
        harmonized = harmonize(composed, alpha, args.harmony, **({"background": bg} if args.harmony == "iclight" else {}))
        harmonized = match_camera(harmonized, alpha) if args.camera else harmonized
        final = add_shadow(add_reflection(harmonized, alpha, args.reflection), alpha, args.shadow, tilt=tilt)

        panels = [(depth_preview(depth, anchor), f"Depth {depth[anchor[1], anchor[0]]:.1f} m at the base"),
                  (composed, "Composed (no harmony)"), (harmonized, f"Harmonized ({args.harmony})")]
        if args.shadow != "none":
            panels.append((final, f"+ shadow ({args.shadow})"))
        figure = labeled_grid(panels)
        for name, img in ((f"{stem}.jpg", composed), (f"{stem}_mask.png", alpha), (f"{stem}_harmonized.jpg", harmonized),
                          (f"{stem}_final.jpg", final), (f"{stem}_comparison.jpg", figure)):
            write_image(args.out_dir / name, img)
        print(f"    saved {stem}*.jpg + {stem}_mask.png in {args.out_dir}")
        outputs.append({"stem": stem, "source": source, "inpainted": inpainted, "composed": composed,
                        "final": final, "alpha": alpha})
        if not args.no_show:
            show(f"Pipeline - {stem}", figure)
    return outputs


if __name__ == "__main__":
    run(build_parser().parse_args())
