"""
Module 1 + 2: builds the object library used by ar_objects.py.

Every museum photo listed in data/objects/catalog.json is cut out once with the Module 2
pipeline - YOLOv8m + SAM 2.1 segmentation, matting and edge decontamination - and saved next to
the catalog as an RGBA PNG cropped to the object, at the photo's full resolution (the photos are at
most 1600 px, 2400 px for the largest objects - held close to a Full HD camera they are drawn up to
~1800 px tall - and 4000 px for the swords, ~3500 px wide across the palms). The live demo then only
resizes and composites these cut-outs.

catalog.json gives, per object, the photo (in photos/), its real height in metres - the one thing
the photo cannot tell - and optionally "class": the YOLO (COCO) class to prefer (default "vase";
another class is used if YOLO finds no vase, and an object YOLO is not sure of at all - a statue,
a helmet, a mask - is cut out by SAM from the largest of its weak detections), and "box":
[x1, y1, x2, y2], where the object is, in fractions of the photo's width and height, for an object
YOLO finds only part of (the celestial globe: the sphere, not the Pegasus and the base under it) -
SAM then cuts it out from that box. "handles": false marks an object without handles (a bust, a
sword): gripping hands then hold it as open hands do, instead of at the widest row of its upper
half; a number puts the grip at that fraction of the height from the top (a dagger's or a war
hammer's grip). For weapons: "width_m" instead of "height_m" gives the real length of an object
lying lengthwise (a sword), "hands": 1 or 2 says which hands hold it whatever its height (a
dagger in one hand, a sword across both palms), "rotate": 90 / 180 / 270 turns the cut-out that
many degrees anticlockwise (a sword photographed upright, laid down; the photo stays as it is).
Add an entry and re-run to extend the library. Also saves catalog_preview.jpg: all objects at one
scale - their real sizes - in rows.

Usage:
    python build_catalog.py            # cut out the objects that have no PNG yet
    python build_catalog.py --force    # redo every cut-out
"""

import argparse

import cv2
import numpy as np

from common.composition import dilate, inpaint
from common.handheld import CATALOG_DIR, catalog_entries, load_catalog, sprite_path
from common.imaging import labeled_grid, read_image, write_image
from common.paths import IMAGES_DIR
from common.segmentation import extract_foreground, load_yolo, refine_mask, sam_mask, segment

MAX_HEIGHT = 2400  # px: the tallest cut-out (photos/CREDITS.md) - every pixel of the painting is kept
HOLE_SHARE = 0.0005  # holes in the mask smaller than this share of the object are SAM's misses, filled
CAPTION_H = 44  # px: labeled_grid's caption bar
MATTING_PX = 1600  # the matting's soft edge is that of a photo this large, also for larger ones: a 4000 px
                   # photo's (16 px) is wider than a sword's blade, which then turned partly see-through


def any_object_box(yolo, image: np.ndarray) -> list[float]:
    """The box of the largest detection of any class down to 5% confidence: a museum photo shows one
    object, and SAM only needs to know roughly where it is."""
    result = yolo(image, verbose=False, conf=0.05)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("Nothing detected even at 5% confidence: give the object's \"box\" in the catalog.")
    boxes = result.boxes.xyxy.cpu().numpy()
    return boxes[int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))].tolist()


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    """The mask with its holes smaller than HOLE_SHARE of the object filled: SAM sometimes leaves a
    few pixels inside an object out (a dark spot of its decoration), which the matting turns into a
    see-through spot; real openings - under a handle, inside a ring - are far larger."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    x, y, w, h, area = stats.T
    inside = (x > 0) & (y > 0) & (x + w < mask.shape[1]) & (y + h < mask.shape[0])  # not the outside
    small = inside & (area < HOLE_SHARE * np.count_nonzero(mask))
    small[0] = False  # label 0: the object itself
    return np.where(small[labels], 255, mask).astype(mask.dtype)


def cut_out(yolo, image: np.ndarray, prefer: str = "vase", box: list[float] | None = None, rotate: int = 0) -> np.ndarray:
    if box is not None:  # the catalog says where the object is: SAM from that box
        h, w = image.shape[:2]
        box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        print(f"  SAM from the catalog's box {[round(v) for v in box]}")
    else:
        try:
            mask, _ = segment(yolo, image, prefer)
        except RuntimeError:  # nothing YOLO is sure of (an object of no YOLO class): SAM from a weaker box
            box = any_object_box(yolo, image)
            print(f"  No confident YOLO detection: SAM from the largest weak one, box {[round(v) for v in box]}")
    if box is not None:
        mask = sam_mask(image, box)
        if mask is None or not mask.any():
            raise RuntimeError("SAM found no object in that box.")
        mask = refine_mask(mask)
    mask = fill_small_holes(mask)
    background = inpaint(image, dilate(mask, 7), "telea")  # what the matting removes from the edges
    sprite = extract_foreground(image, mask, background, pad=2, radius=max(2, round(0.004 * min(MATTING_PX, max(image.shape[:2])))))
    if rotate:  # turned anticlockwise, after the cut-out: the photo itself stays as it is
        sprite = np.ascontiguousarray(np.rot90(sprite, rotate // 90))
    if sprite.shape[0] > MAX_HEIGHT:
        sprite = cv2.resize(sprite, (round(sprite.shape[1] * MAX_HEIGHT / sprite.shape[0]), MAX_HEIGHT),
                            interpolation=cv2.INTER_AREA)
    return sprite


def preview(catalog, tallest_px: int = 480, row_px: int = 2600) -> np.ndarray:
    """The objects on a grey floor at one scale - their real sizes, the tallest tallest_px high - those for
    one hand first, then those for two, each smallest first, in rows of up to about row_px; each captioned
    with the size the catalog gives."""
    tallest = max(a.height_m for a in catalog)
    sprites = []
    for art in sorted(catalog, key=lambda a: (not a.one_hand, a.height_m)):
        h = max(8, round(tallest_px * art.height_m / tallest))
        sprites.append((art, cv2.resize(art.sprite, (max(1, round(art.sprite.shape[1] * h / art.sprite.shape[0])), h),
                                        interpolation=cv2.INTER_AREA).astype(np.float32)))

    def tile_w(sprite: np.ndarray) -> int:  # with room for the caption
        return max(sprite.shape[1] + 20, 90)

    groups = []
    for one_hand in (True, False):  # those for two hands from a new row; rows of about equal width
        items = [item for item in sprites if item[0].one_hand == one_hand]
        total = sum(tile_w(s) for _, s in items)
        target, group, width = total / max(1, -(-total // row_px)), [], 0
        for item in items:
            if group and width + tile_w(item[1]) / 2 > target:
                groups.append(group)
                group, width = [], 0
            group.append(item)
            width += tile_w(item[1])
        if group:
            groups.append(group)
    rows = []
    for group in groups:
        floor = max(s.shape[0] for _, s in group)  # the row as tall as its tallest object
        panels = []
        for art, sprite in group:
            h, w = sprite.shape[:2]
            tile = np.full((floor, tile_w(sprite), 3), 200, np.float32)
            a = sprite[..., 3:] / 255
            roi = tile[floor - h:, (tile.shape[1] - w) // 2:(tile.shape[1] - w) // 2 + w]
            roi += a * (sprite[..., :3] - roi)
            panels.append((tile.astype(np.uint8), f"{100 * art.size_m:.0f} cm"))
        rows.append(labeled_grid(panels, height=floor, label_h=CAPTION_H))
    width = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:  # the narrower row continued: its caption bar black, its floor grey
        pad = np.full((r.shape[0], width - r.shape[1], 3), 200, np.uint8)
        pad[:CAPTION_H] = 0
        padded.append(np.hstack([r, pad]))
    return np.vstack(padded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cut out the museum objects of data/objects/catalog.json")
    parser.add_argument("--force", action="store_true", help="Redo the cut-outs that already exist")
    args = parser.parse_args()

    yolo = None
    for entry in catalog_entries():
        path = sprite_path(entry)
        if path.exists() and not args.force:
            continue
        yolo = yolo or load_yolo("m")
        sprite = cut_out(yolo, read_image(IMAGES_DIR / entry["photo"]), entry.get("class", "vase"), entry.get("box"),
                         entry.get("rotate", 0))
        write_image(path, sprite)
        print(f"  {path.name}: {sprite.shape[1]}x{sprite.shape[0]} px, "
              + (f"{entry['height_m']} m tall" if "height_m" in entry else f"{entry['width_m']} m long"))
    catalog = load_catalog()
    for art in catalog:
        print(f"{art.name:<28} {'one hand ' if art.one_hand else 'two hands'} height {art.height_m:.3f} m, "
              f"belly {art.belly_width_m:.3f} m wide, "
              + (f"grip at {art.handles:.2f} of its height, {art.handles_width_m:.3f} m wide" if art.handles is not None
                 else "no handles"))
    print("Saved:", write_image(CATALOG_DIR / "catalog_preview.jpg", preview(catalog)))


if __name__ == "__main__":
    main()
