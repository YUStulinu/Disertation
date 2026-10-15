"""
Illumination-aware harmonization: Intrinsic Harmonization (Careaga, Miangoleh & Aksoy,
SIGGRAPH Asia 2023, github.com/compphoto/IntrinsicCompositing, academic-use licence).

Colour-only harmonization cannot change how an object is SHADED. This method splits the
composite into albedo (the object's own colour) and shading, harmonizes the albedo,
estimates the scene's lighting (spherical harmonics) from the background's shading and
surface normals, and re-shades the object as if lit by that light - so the light direction
on the object follows the scene. The networks run at `size` px; the change they make is
transferred to the full-resolution image as a smooth gain map, keeping every fine detail.
"""

from pathlib import Path

import cv2
import numpy as np

_BOOSTED = "https://github.com/CCareaga/BoostingMonocularDepth/releases/download/v0.0.1/"
WGET_WEIGHTS = {  # weights the packages download with `wget` (absent on Windows), relative to the torch hub cache
    "midas/model.pt": _BOOSTED + "model.pt",
    "mergemodel/latest_net_G.pth": _BOOSTED + "latest_net_G.pth",
    "omnidata/omnidata_dpt_normal_v2.ckpt": "https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1",
    "albedo_harmonization/albedo_paper_weights.pth":
        "https://github.com/compphoto/IntrinsicCompositing/releases/download/1.0.0/albedo_paper_weights.pth"}
_models: dict = {}


def _fetch_weights() -> None:
    import torch

    for name, url in WGET_WEIGHTS.items():
        path = Path(torch.hub.get_dir()) / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(url, str(path))


def _omni_normals():
    """Omnidata DPT-Hybrid surface normals. The checkpoint is a PyTorch Lightning file: it is read in
    weights-only mode with inert stand-ins for the Lightning objects stored beside the weights, so
    unpickling never runs code from the file (the package's own loader needs weights_only=False)."""
    import pickle
    import re

    import torch
    from omnidata_tools.model_util import OMNIDATA_NORMALS_WEIGHTS_PATH
    from omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel

    stand_ins = []
    while True:
        try:
            with torch.serialization.safe_globals(stand_ins):
                checkpoint = torch.load(OMNIDATA_NORMALS_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            break
        except pickle.UnpicklingError as error:
            name = re.search(r"GLOBAL (\S+) was not an allowed global", str(error))
            if name is None or len(stand_ins) > 20:
                raise
            inert = type(name[1].rsplit(".", 1)[-1], (), {"__init__": lambda self, *args, **kwargs: None})
            stand_ins.append((inert, name[1]))
    model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
    model.load_state_dict({k.removeprefix("model."): v for k, v in checkpoint["state_dict"].items()})
    return model.to("cuda").eval()



def unload() -> None:
    """Free the GPU memory of these networks (they are reloaded on the next call)."""
    if _models:
        import gc

        import torch

        _models.clear()
        gc.collect()
        torch.cuda.empty_cache()

def _load():
    if not _models:
        _fetch_weights()
        from boosted_depth.depth_util import create_depth_models
        from intrinsic.model_util import load_models
        from intrinsic_compositing.albedo.pipeline import load_albedo_harmonizer
        from intrinsic_compositing.shading.pipeline import load_reshading_model

        _models.update(depth=create_depth_models(), normals=_omni_normals(), intrinsic=load_models("paper_weights"),
                       albedo=load_albedo_harmonizer(), reshading=load_reshading_model("further_trained"))
    return _models


def _resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if size[0] < img.shape[1] else cv2.INTER_LINEAR)


def intrinsic(image: np.ndarray, alpha: np.ndarray, size: int = 768, albedo: bool = True) -> np.ndarray:
    """albedo=False keeps the object's own colours and only re-shades it (the light direction)."""
    from boosted_depth.depth_util import get_depth
    from chrislib.general import round_32, uninvert
    from chrislib.normal_util import get_omni_normals
    from intrinsic.pipeline import run_pipeline
    from intrinsic_compositing.albedo.pipeline import harmonize_albedo
    from intrinsic_compositing.shading.pipeline import compute_reshading, get_light_coeffs

    from common.composition import blend

    m = _load()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    mask = alpha.astype(np.float32) / 255
    h, w = rgb.shape[:2]

    # 1. scene lighting from the background's shading and normals (at 512 px, as in the paper)
    s = 512 / max(h, w)
    small = _resize(rgb, (round(w * s), round(h * s)))
    shading = run_pipeline(m["intrinsic"], small ** 2.2, resize_conf=0.0, maintain_size=True, linear=True)["inv_shading"]
    coeffs, _ = get_light_coeffs(shading, get_omni_normals(m["normals"], small), small)

    # 2. normals of the composite, with a detailed estimate of the object from a tight crop
    s = min(1.0, size / max(h, w))
    work = (round_32(w * s), round_32(h * s))
    img, msk = _resize(rgb, work), _resize(mask, work)
    normals = get_omni_normals(m["normals"], img)
    ys, xs = np.nonzero(msk > 0.5)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[int(y1 / s):int(y2 / s), int(x1 / s):int(x2 / s)]
    c = 1024 / max(crop.shape[:2])
    obj_normals = get_omni_normals(m["normals"], _resize(crop, (round(crop.shape[1] * c), round(crop.shape[0] * c))))
    patch = np.zeros_like(normals)
    patch[y1:y2, x1:x2] = _resize(obj_normals, (x2 - x1, y2 - y1))
    normals = patch * msk[..., None] + normals * (1 - msk[..., None])

    # 3. harmonized albedo x re-shading under the scene light
    inv_shd = run_pipeline(m["intrinsic"], img ** 2.2, resize_conf=0.0, maintain_size=True, linear=True)["inv_shading"]
    shading = uninvert(inv_shd)[..., None]
    alb = harmonize_albedo(img, msk, inv_shd, m["albedo"]) ** 2.2 if albedo else img ** 2.2 / shading.clip(1e-4)
    result = compute_reshading(alb * shading, msk, inv_shd, get_depth(img, m["depth"]), normals, alb, coeffs,
                               m["reshading"])["composite"]

    # 4. transfer the change to full resolution as a smooth multiplicative gain inside the object
    gain = cv2.GaussianBlur((np.clip(result, 0, 1) + 0.02) / (img + 0.02), (0, 0), 1.5)
    out = np.clip(rgb * cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)
    out = cv2.cvtColor((out * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return blend(image, out, alpha)
