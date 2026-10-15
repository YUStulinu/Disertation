"""
Module 2 - Step 5: harmonization - make the pasted object match the scene.

Runs the chosen methods (see common/harmonization.py) on one or more composites
and saves each result plus a comparison figure:
  ensemble   - mean of the three networks below (default, best on the benchmark)
  harmonizer - Harmonizer (Ke et al., ECCV 2022), trained on iHarmony4
  pctnet     - PCT-Net ViT (Guerreiro et al., CVPR 2023), pctnet_cnn - its CNN version
  vgg        - VGG16 feature-statistics baseline (no training), --layer = deepest layer used
  reinhard / histogram - classical colour transfer baselines;  none - the raw composite
  intrinsic  - Intrinsic Harmonization (SIGGRAPH Asia 2023): albedo + re-shading under the scene light
  reshade    - ensemble colours + the re-shading step of Intrinsic Harmonization
  iclight    - IC-Light (ICLR 2025) diffusion relighting, only the lighting change is kept
The figure is also saved cropped around the object (_comparison_detail.jpg);
--ablation adds the VGG16 depth figure (relu1_2 ... relu4_3).

Each composite needs its placed mask (<name>_mask.png), saved by compose.py /
pipeline.py; --all processes every such composite in data/composition_results.

Usage:
    python harmonize.py --composed data/composition_results/photo_on_bg.jpg [--methods harmonizer reinhard]
    python harmonize.py --all --ablation
"""

import argparse
from pathlib import Path

from common.harmonization import METHOD_NAMES, METHODS, VGG_LAYERS, harmonize
from common.imaging import crop_object, labeled_grid, read_composite, show, write_image
from common.paths import DATA_DIR, mask_path_for

OUTPUT_DIR = DATA_DIR / "harmonization_results"


def run(path: Path, methods: list[str], layer: str, ablation: bool):
    """Harmonize one composite with every method; returns the comparison figure."""
    image, alpha = read_composite(path)
    results = {}
    for method in methods:
        print(f"  {method}...")
        results[method] = harmonize(image, alpha, method, **({"layer": layer} if method == "vgg" else {}))
        write_image(OUTPUT_DIR / f"{path.stem}_{method}.jpg", results[method])
    cols = 4 if len(methods) > 4 else None
    figure = labeled_grid([(results[m], METHOD_NAMES[m]) for m in methods], cols)
    print("  Saved:", write_image(OUTPUT_DIR / f"{path.stem}_comparison.jpg", figure))
    detail = labeled_grid([(crop_object(results[m], alpha, 0.35), METHOD_NAMES[m]) for m in methods], cols)
    print("  Saved:", write_image(OUTPUT_DIR / f"{path.stem}_comparison_detail.jpg", detail))
    if ablation:
        depth = [(image, "No harmonization")]
        for name in VGG_LAYERS:  # reuse the main "vgg" result for its own layer
            reused = "vgg" in results and name == layer
            depth.append((results["vgg"] if reused else harmonize(image, alpha, "vgg", layer=name), f"VGG16 up to {name}"))
        print("  Saved:", write_image(OUTPUT_DIR / f"{path.stem}_vgg_depth_ablation.jpg", labeled_grid(depth)))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonization - Module 2 Day 15 / Week 4")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--composed", nargs="+", type=Path, help="Composite image(s) with a <name>_mask.png")
    source.add_argument("--all", action="store_true", help="Every composite in data/composition_results")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--layer", choices=list(VGG_LAYERS), default="relu3_3", help="Deepest VGG16 layer (vgg)")
    parser.add_argument("--ablation", action="store_true", help="Also save the VGG16 depth ablation figure")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    paths = args.composed or [p for p in sorted((DATA_DIR / "composition_results").glob("*.jpg"))
                              if mask_path_for(p).exists()]
    for path in paths:
        print(path.name)
        figure = run(path, args.methods, args.layer, args.ablation)
        if not args.no_show:
            show(f"Harmonization - {path.stem}", figure)


if __name__ == "__main__":
    main()
