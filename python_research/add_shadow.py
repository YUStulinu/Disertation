"""
Module 2 - Week 6 Day 4: add a shadow under the composed object.

A shadow is one of the strongest cues that an object is physically standing
in the scene rather than pasted on. Two modes, both drawn only on the floor
around the object (never on the object itself):
  ellipse - soft blurred contact shadow under the object's base (default)
  cast    - the silhouette laid flat on the floor plane behind the object, in
            the direction of --angle (0 = right, 90 = straight back, 135 = back-left)

The object position comes from the composite's placed mask (<image>_mask.png).

Usage:
    python add_shadow.py --composed data/museum_demo/poza_muzeu_on_sala1_harmonized.jpg [--mode cast --angle 60]
    python add_shadow.py --batch data/museum_demo      # every *_harmonized.jpg (pipeline output before the shadow)
"""

import argparse
from pathlib import Path

from common.composition import add_shadow
from common.imaging import labeled_grid, read_composite, show, write_image
from common.paths import DATA_DIR, mask_path_for

OUTPUT_DIR = DATA_DIR / "shadow_results"


def process(path: Path, args) -> None:
    image, alpha = read_composite(path, mask_path_for(path))
    kw = {"opacity": args.opacity, "blur": args.blur}
    if args.mode == "cast":
        kw |= {"angle_deg": args.angle, "length": args.length}
    result = add_shadow(image, alpha, args.mode, **kw)
    figure = labeled_grid([(image, "Without shadow"), (result, f"With shadow ({args.mode})")])
    print("Saved:", write_image(OUTPUT_DIR / f"{path.stem}_shadow_{args.mode}.jpg", result))
    print("Saved:", write_image(OUTPUT_DIR / f"{path.stem}_shadow_{args.mode}_cmp.jpg", figure))
    if not args.no_show:
        show(f"Shadow - {path.stem}", figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add a shadow to a composite - Week 6 Day 4")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--composed", type=Path, help="One composite (needs <name>_mask.png next to it)")
    source.add_argument("--batch", type=Path, help="Folder: every *_harmonized.jpg with a matching mask")
    parser.add_argument("--mode", default="ellipse", choices=["ellipse", "cast"])
    parser.add_argument("--angle", type=float, default=60.0, help="Cast shadow direction in degrees")
    parser.add_argument("--length", type=float, default=0.35, help="Cast shadow length / object height")
    parser.add_argument("--opacity", type=float, default=0.45, help="Shadow darkness 0-1")
    parser.add_argument("--blur", type=int, help="Edge softness in px (default: proportional to object size)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    paths = [args.composed] if args.composed else sorted(args.batch.glob("*_harmonized.jpg"))
    for path in paths:
        process(path, args)
    if not paths:
        print(f"No *_harmonized.jpg files in {args.batch}.")


if __name__ == "__main__":
    main()
