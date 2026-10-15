"""
Diffusion relighting: IC-Light (Zhang, Rao & Agrawala, ICLR 2025), the background-conditioned
model on Stable Diffusion 1.5 (Realistic Vision 5.1). Given the object on a neutral grey canvas
and the background, it generates the object lit by the background's light.

A diffusion model can also change fine details (e.g. a vase's painting), so only the change in
LIGHTING is kept: the ratio of the blurred relit object to the blurred original is applied as a
smooth gain to the full-resolution composite. Runs on a 6 GB GPU in fp16 (VAE in fp32: Turing
GPUs have no bfloat16), about 10-30 s per image on an RTX 2060; weights (~6 GB) download once.
"""

import cv2
import numpy as np

SD15 = "stablediffusionapi/realistic-vision-v51"
_pipe: list = []



def unload() -> None:
    """Free the GPU memory of these networks (they are reloaded on the next call)."""
    if _pipe:
        import gc

        import torch

        _pipe.clear()
        gc.collect()
        torch.cuda.empty_cache()

def _load():
    if not _pipe:
        import safetensors.torch as sf
        import torch
        from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from diffusers.models.attention_processor import AttnProcessor2_0
        from huggingface_hub import hf_hub_download
        from transformers import CLIPTextModel, CLIPTokenizer

        unet = UNet2DConditionModel.from_pretrained(SD15, subfolder="unet")
        with torch.no_grad():  # 12 input channels: noise latent + object latent + background latent
            conv = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
            conv.weight.zero_()
            conv.weight[:, :4].copy_(unet.conv_in.weight)
            conv.bias = unet.conv_in.bias
            unet.conv_in = conv
        offset = sf.load_file(hf_hub_download("lllyasviel/ic-light", "iclight_sd15_fbc.safetensors"))
        unet.load_state_dict({k: v + offset[k] for k, v in unet.state_dict().items()}, strict=True)
        forward = unet.forward

        def hooked(sample, timestep, encoder_hidden_states, **kw):  # concatenate the conditions to the noise
            cond = kw.pop("cross_attention_kwargs")["concat_conds"].to(sample)
            cond = torch.cat([cond] * (sample.shape[0] // cond.shape[0]))
            return forward(torch.cat([sample, cond], dim=1), timestep, encoder_hidden_states, **kw)

        unet.forward = hooked
        vae = AutoencoderKL.from_pretrained(SD15, subfolder="vae").to("cuda", torch.float32)
        unet = unet.to("cuda", torch.float16)
        for model in (unet, vae):
            model.set_attn_processor(AttnProcessor2_0())
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012,
                                                algorithm_type="sde-dpmsolver++", use_karras_sigmas=True, steps_offset=1)
        _pipe.append(StableDiffusionPipeline(
            vae=vae, unet=unet, scheduler=scheduler, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False, image_encoder=None, tokenizer=CLIPTokenizer.from_pretrained(SD15, subfolder="tokenizer"),
            text_encoder=CLIPTextModel.from_pretrained(SD15, subfolder="text_encoder").to("cuda", torch.float16)))
    return _pipe[0]


def relight(image: np.ndarray, alpha: np.ndarray, background: np.ndarray | None = None, prompt: str = "an object",
            width: int = 768, steps: int = 25, seed: int = 0) -> np.ndarray:
    import torch

    from common.composition import blend

    pipe = _load()
    h, w = image.shape[:2]
    if background is None:  # what is behind the object: inpaint it out of the composite
        background = cv2.inpaint(image, cv2.dilate((alpha > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)), 5, cv2.INPAINT_TELEA)
    size = (width // 64 * 64, round(h * width / w / 64) * 64)
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    rgb = cv2.cvtColor(cv2.resize(image, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    fg = 127 + (rgb - 127) * a  # the object on neutral grey
    bg = cv2.cvtColor(cv2.resize(background, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32)
    x = torch.from_numpy(np.stack([fg, bg])).permute(0, 3, 1, 2).cuda() / 127.5 - 1
    with torch.inference_mode():
        lat = pipe.vae.encode(x).latent_dist.mode() * pipe.vae.config.scaling_factor
        cond = torch.cat([lat[0:1], lat[1:2]], dim=1)
        out = pipe(prompt=f"{prompt}, natural lighting, best quality", negative_prompt="lowres, bad anatomy, worst quality",
                   width=size[0], height=size[1], num_inference_steps=steps, guidance_scale=2.0, output_type="latent",
                   generator=torch.Generator("cuda").manual_seed(seed), cross_attention_kwargs={"concat_conds": cond}).images
        pixels = pipe.vae.decode(out.float() / pipe.vae.config.scaling_factor).sample
    relit = ((pixels[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).clip(0, 255)

    # keep only the lighting change: smooth gain = blurred relit / blurred original, inside the object
    ys = np.nonzero(a[..., 0] > 0.5)[0]
    sigma = max(1.5, 0.04 * np.ptp(ys)) if len(ys) else 2.0
    weight = cv2.GaussianBlur(a[..., 0], (0, 0), sigma)[..., None] + 1e-3
    smooth = lambda img: cv2.GaussianBlur(img * a, (0, 0), sigma) / weight  # blur that ignores the background
    gain = np.clip((smooth(relit) + 2) / (smooth(rgb) + 2), 0.5, 2.0)
    gain = cv2.resize(cv2.cvtColor(gain.astype(np.float32), cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_LINEAR)
    return blend(image, np.clip(image * gain + 0.5, 0, 255).astype(np.uint8), alpha)
