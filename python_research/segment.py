"""
Module 2 - Steps 1-2: segmentation and foreground extraction (YOLOv8-seg).

Draws every detected object, picks one (--index, else the most confident one
of --class, else the most confident overall), refines its mask with SAM 2.1
(prompted with the YOLO box), mattes the edge and saves:
  segmentation_results/<stem>_segmented.jpg : all detections
  foreground_results/<stem>_foreground.png  : tight RGBA cut-out for compose.py
  foreground_results/<stem>_mask.png        : binary mask in SOURCE coordinates (for inpaint.py)
  foreground_results/<stem>_preview.jpg     : the object on black

With --model n m both YOLOv8 sizes run (nano ~7 MB, medium ~52 MB): side-by-side
figure with the SAM-refined mask, IoU/Dice agreement of the masks and, with --gt-mask,
IoU/Dice against a hand-annotated mask. The cut-out comes from the last model.

Usage:
    python segment.py --image photo.jpg --class person
    python segment.py --image museum.jpg --class vase --model n m [--gt-mask vase_gt.png]
    python segment.py --url https://example.com/photo.jpg
    python segment.py --list-classes
"""

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common.composition import inpaint
from common.imaging import labeled_grid, read_image, read_mask, show, write_image
from common.paths import DATA_DIR
from common.segmentation import (describe, draw_detections, extract_foreground, iou_dice, load_yolo, pick_detection,
                                 refine_mask, sam_mask, segment)


def download_image(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as response:
        img = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"The URL did not return a readable image: {url}")
    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation + foreground extraction - Module 2 Days 11-12")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="Local image")
    source.add_argument("--url", help="Image URL")
    parser.add_argument("--class", dest="cls", help="Object to extract (e.g. person, vase)")
    parser.add_argument("--index", type=int, default=-1, help="Extract the Nth detection instead")
    parser.add_argument("--model", nargs="+", choices=["n", "m"], default=["n"], help="YOLOv8 size(s): n, m or n m")
    parser.add_argument("--sam", action=argparse.BooleanOptionalAction, default=True, help="Refine the mask with SAM 2.1")
    parser.add_argument("--matting", action=argparse.BooleanOptionalAction, default=True, help="Soft, decontaminated edges")
    parser.add_argument("--gt-mask", help="Hand-annotated binary mask, for IoU / Dice")
    parser.add_argument("--list-classes", action="store_true", help="Print the 80 COCO classes and exit")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    if args.list_classes:
        print("\n".join(f"  {i:3d}  {name}" for i, name in sorted(load_yolo("n").names.items())))
        return
    if not (args.image or args.url):
        parser.error("Provide --image <path> or --url <url>.")
    image = download_image(args.url) if args.url else read_image(args.image)
    stem = "from_url" if args.url else Path(args.image).stem
    gt = read_mask(args.gt_mask, image.shape[:2]) if args.gt_mask else None

    panels, masks = [(image, "Original")], {}
    for size in args.model:
        print(f"YOLOv8{size}-seg:")
        masks[size], result = segment(load_yolo(size), image, args.cls, args.index, sam=False)
        dets = describe(result)
        print("  detections: " + ", ".join(f"{name} {conf:.2f}" for name, conf in dets))
        panels.append((draw_detections(image, result), f"YOLOv8{size}-seg: {len(dets)} det."))
    mask = masks[args.model[-1]]
    if args.sam:  # refine the last model's chosen detection
        refined = sam_mask(image, result.boxes[pick_detection(result, args.cls, args.index)].xyxy[0].tolist())
        if refined is not None and refined.any():
            mask = refine_mask(refined)
            print("SAM 2.1 refinement: IoU with the YOLO mask = {:.3f}".format(iou_dice(mask, masks[args.model[-1]])[0]))
            tinted = image.copy()
            tinted[mask > 0] = (0.5 * tinted[mask > 0] + (0, 127, 0)).astype(np.uint8)
            panels.append((tinted, "SAM 2.1 mask"))
    if gt is not None:
        for name, m in [*masks.items(), ("final", mask)]:
            iou, dice = iou_dice(m, gt)
            print(f"  {name} vs ground truth: IoU={iou:.3f}  Dice={dice:.3f}")
    if len(masks) == 2:
        print("Agreement n vs m: IoU={:.3f}  Dice={:.3f}".format(*iou_dice(masks["n"], masks["m"])))
        for size, m in masks.items():
            write_image(DATA_DIR / "yolo_comparison" / f"{stem}_mask_{size}.png", m)
        path = DATA_DIR / "yolo_comparison" / f"{stem}_n_vs_m.jpg"
    else:
        path = DATA_DIR / "segmentation_results" / f"{stem}_segmented.jpg"
    figure = labeled_grid(panels)
    print("Saved:", write_image(path, figure))

    out = DATA_DIR / "foreground_results"
    cutout = extract_foreground(image, mask, inpaint(image, mask, "telea"), args.matting)
    for suffix, img in (("foreground.png", cutout), ("mask.png", mask), ("preview.jpg", image * (mask[..., None] > 0))):
        print("Saved:", write_image(out / f"{stem}_{suffix}", img))
    if not args.no_show:
        show("Segmentation", figure)


if __name__ == "__main__":
    main()
