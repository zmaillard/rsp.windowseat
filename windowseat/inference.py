import functools
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torchvision
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image
from tile import TilingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def visualize(
    file_prefix: str,
    input_hwc: np.ndarray,
    pred_hwc: np.ndarray,
    output_dir: str,
) -> None:
    pred_hwc = pred_hwc.clip(0, 1)
    pred_uint8 = (pred_hwc * 255).round().astype(np.uint8)
    input_hwc = np.asarray(input_hwc, dtype=np.float32)
    if input_hwc.max() > 1.0:
        input_hwc = input_hwc / 255.0

    save_prediction_only(
        file_prefix=file_prefix,
        pred_uint8=pred_uint8,
        output_dir=output_dir,
    )


def save_prediction_only(
    file_prefix: str,
    pred_uint8: np.ndarray,
    output_dir: str,
) -> None:
    imageio.imwrite(
        os.path.join(output_dir, f"{file_prefix}_windowseat_output.png"),
        pred_uint8,
        plugin="pillow",
    )


def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.size(0) == B:
        return t
    if t.size(0) == 1 and B > 1:
        return t.expand(B, *t.shape[1:])
    if t.size(0) > B:
        return t[:B]
    reps = (B + t.size(0) - 1) // t.size(0)
    return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]


def flow_step(
    model_input: torch.Tensor,
    transformer: QwenImageTransformer2DModel,
    vae: AutoencoderKLQwenImage,
    embeds_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    prompt_embeds = embeds_dict["prompt_embeds"]  # [N_ctx, L, D]
    prompt_mask = embeds_dict["prompt_mask"]  # [N_ctx, L]

    if prompt_mask.dtype != torch.bool:
        prompt_mask = prompt_mask > 0

    # Accept [B, C, 1, H, W] or [B, C, H, W]
    if model_input.ndim == 5 and model_input.shape[2] == 1:
        model_input_4d = model_input[:, :, 0]  # [B, C, H, W]
    elif model_input.ndim == 4:
        model_input_4d = model_input
    else:
        raise ValueError(f"Unexpected lat_encoding shape: {model_input.shape}")

    B, C, H, W = model_input_4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = _match_batch(prompt_embeds, B).to(
        device=device, dtype=torch.bfloat16, non_blocking=True
    )  # [B, L, D]

    prompt_mask = _match_batch(prompt_mask, B).to(
        device=device, dtype=torch.bool, non_blocking=True
    )  # [B, L]

    num_channels_latents = C
    packed_model_input = QwenImageEditPipeline._pack_latents(
        model_input_4d,
        batch_size=B,
        num_channels_latents=num_channels_latents,
        height=H,
        width=W,
    )  # [B, N_patches, C * 4], where N_patches = (H // 2) * (W // 2)
    packed_model_input = packed_model_input.to(torch.bfloat16)

    t_const = 499
    timestep = torch.full(
        (B,),
        float(t_const),
        device=device,
        dtype=torch.bfloat16,
    )
    timestep = timestep / 1000.0

    h_img = H // 2
    w_img = W // 2

    img_shapes = [[(1, h_img, w_img)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist() if prompt_mask is not None else None

    if getattr(transformer, "attention_kwargs", None) is None:
        attention_kwargs = {}
    else:
        attention_kwargs = transformer.attention_kwargs

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_model_input,  # [B, N_patches, C*4]
            timestep=timestep,  # [B], float / 1000
            encoder_hidden_states=prompt_embeds,  # [B, L, D]
            encoder_hidden_states_mask=prompt_mask,  # [B, L]
            img_shapes=img_shapes,  # single stream per batch
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]  # [B, N_patches, C*4]

    temperal_downsample = vae.config.get("temperal_downsample", None)
    if temperal_downsample is not None:
        vae_scale_factor = 2 ** len(temperal_downsample)
    else:
        vae_scale_factor = 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred,
        height=H * vae_scale_factor,  # H, W here are latent H,W from encode
        width=W * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )  # [B, C, 1, H_lat, W_lat]

    latent_output = model_input.to(vae.dtype) - model_pred.to(vae.dtype)

    return latent_output


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=latents.dtype
    )
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(
        vae.config.latents_std, device=latents.device, dtype=latents.dtype
    )
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    out = out.sample[:, :, 0]
    return out


def encode(image: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    image = image.to(device=vae.device, dtype=vae.dtype)
    out = vae.encode(image.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(
        vae.config.latents_mean, device=out.device, dtype=out.dtype
    )
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(
        vae.config.latents_std, device=out.device, dtype=out.dtype
    )
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    out = (out - latents_mean) * latents_std_inv
    return out


@torch.no_grad()
def validate_single_dataset(
    vae: AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    embeds_dict: dict[str, torch.Tensor],
    data_loader: DataLoader,
    save_to_dir: str,
):
    preds = []

    for i, batch in enumerate(
        tqdm(data_loader, desc=f"Reflection Removal Progress"),
        start=1,
    ):
        batch["out"] = {}
        with torch.no_grad():
            latents = encode(batch["input_norm"], vae)
            latents = flow_step(latents, transformer, vae, embeds_dict)
            batch["out"]["pixel_pred"] = decode(latents, vae)

        for b in range(len(batch["idx"])):
            preds.append(
                {
                    "file": batch["line"][0][b],
                    # [x0, y0, x1, y1] tuple for the tile
                    "tile_info": [batch["tile_info"][i][b] for i in range(4)],
                    # Shape 1, 3, H, W, torch tensor in range -1 to 1
                    "pred": batch["out"]["pixel_pred"][b].to("cpu"),
                }
            )

            if batch["is_last_tile"][b]:
                # Stitch predictions together
                W = max(int(t["tile_info"][2]) for t in preds)
                H = max(int(t["tile_info"][3]) for t in preds)

                acc = torch.zeros(3, H, W, dtype=torch.float32)
                wsum = torch.zeros(H, W, dtype=torch.float32)

                for t in preds:
                    tile_info = [t["tile_info"][i] for i in range(4)]
                    x0, y0, x1, y1 = map(int, tile_info)
                    tile = t["pred"].squeeze(0).float()  # [3, h, w], [-1,1]

                    h, w = tile.shape[-2:]
                    tH, tW = (y1 - y0), (x1 - x0)
                    if (h != tH) or (w != tW):
                        tile = _lanczos_resize_chw(tile, (tH, tW))
                        h, w = tH, tW

                    # triangular window for the tile
                    # fmt: off
                    wx = 1 - (2 * torch.arange(w, dtype=torch.float32) / (max(w - 1, 1)) - 1).abs()
                    wy = 1 - (2 * torch.arange(h, dtype=torch.float32) / (max(h - 1, 1)) - 1).abs()
                    # fmt: on
                    w2 = (wy[:, None] * wx[None, :]).clamp_min(1e-3)
                    acc[:, y0:y1, x0:x1] += tile * w2
                    wsum[y0:y1, x0:x1] += w2
                stitched = (acc / wsum.clamp_min(1e-6)).unsqueeze(
                    0
                )  # [1,3,H,W], [-1,1]

                # Lanczos resize to gt_orig shape
                orig_H, orig_W = (
                    batch["meta"]["orig_res"][0][b].item(),
                    batch["meta"]["orig_res"][1][b].item(),
                )

                x = stitched.squeeze(0)
                x01 = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
                device = x01.device

                pil = torchvision.transforms.functional.to_pil_image(x01.cpu())
                pil_resized = pil.resize((orig_W, orig_H), resample=Image.LANCZOS)
                pred_ts = torchvision.transforms.functional.to_tensor(pil_resized).to(
                    device
                )  # [3,H,W], [0,1]
                pred = pred_ts.cpu().numpy()
                preds = []
            else:
                continue

            pred_ts = torch.from_numpy(pred).to(device)  # [3,H,W]
            scene_path = batch["line"][0][b]
            scene_name = scene_path.split("/")[-1][:-4]

            # Load original input image (CHW, uint8 in [0,255])
            input_chw = read_rgb_file(scene_path)
            input_hwc = (
                np.transpose(input_chw, (1, 2, 0)).astype(np.float32) / 255.0
            )  # [H,W,3], [0,1]

            pred_hwc = np.transpose(pred, (1, 2, 0))
            if input_hwc.shape[:2] != pred_hwc.shape[:2]:
                pil_pred = Image.fromarray(
                    (pred_hwc.clip(0, 1) * 255).round().astype(np.uint8)
                )
                H_in, W_in = input_hwc.shape[:2]
                pil_pred = pil_pred.resize((W_in, H_in), resample=Image.LANCZOS)
                pred_hwc = (np.array(pil_pred, dtype=np.uint8) / 255.0).clip(0, 1)

            visualize(
                file_prefix=scene_name,
                input_hwc=input_hwc,
                pred_hwc=pred_hwc,
                output_dir=save_to_dir,
            )

    return


def read_rgb_file(rgb_path) -> np.ndarray:
    img = Image.open(rgb_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)  # [H, W, 3]
    return arr.transpose(2, 0, 1)  # [3, H, W]


def load_rgb_data(rgb_path, key_prefix="input"):
    rgb = read_rgb_file(rgb_path)
    rgb_norm = rgb / 255.0 * 2.0 - 1.0
    outputs = {
        f"{key_prefix}_int": torch.from_numpy(rgb).int(),
        f"{key_prefix}_norm": torch.from_numpy(rgb_norm),
    }
    return outputs


def read_rgb_image(sample):
    column = 0
    name = "input"

    img_path = sample["line"][column]
    img = load_rgb_data(img_path, name)
    sample.update(img)
    sample.setdefault("meta", {})
    sample["meta"]["orig_res"] = [
        sample[name + "_norm"].shape[-2],
        sample[name + "_norm"].shape[-1],
    ]


def read_scalars(sample):
    scalar_dict = {"tile_info": 1, "is_last_tile": 2}
    for name, col in scalar_dict.items():
        sample[name] = sample["line"][col]


def data_transform(sample, processing_resolution=None):
    read_scalars(sample)
    read_rgb_image(sample)
    tile(sample, processing_resolution)


def tile(sample, processing_resolution: int):
    x0, y0, x1, y1 = map(int, sample["tile_info"])
    processing_width = x1 - x0
    processing_height = y1 - y0

    # Reshape input while keeping aspect ratio
    H, W = sample["input_norm"].shape[-2:]
    if W < processing_width or H < processing_height:
        min_side = min(W, H)
        scale_ratio = processing_width / min_side
        W = round(scale_ratio * W)
        H = round(scale_ratio * H)

    reshape(sample, height=H, width=W)
    sample["input_int"] = sample["input_int"][:, y0:y1, x0:x1]
    sample["input_norm"] = sample["input_norm"][:, y0:y1, x0:x1]
    reshape(sample, height=processing_resolution, width=processing_resolution)


def _lanczos_resize_chw(x, out_hw):
    H_out, W_out = map(int, out_hw)

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        dev = x.device
        arr = x.detach().cpu().numpy()
    else:
        arr = x

    assert isinstance(arr, np.ndarray) and arr.ndim == 3, "expect CHW"
    chw = arr.astype(np.float32, copy=False)
    C, _, _ = chw.shape

    out_chw = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        ch = chw[c]
        img = Image.fromarray(ch).convert("F")
        img = img.resize((W_out, H_out), resample=Image.LANCZOS)
        out_chw[c] = np.asarray(img, dtype=np.float32)

    if is_torch:
        return torch.from_numpy(out_chw).to(dev)
    return out_chw


def reshape(sample, height, width):
    Ht, Wt = height, width
    for k, v in list(sample.items()):
        if not (torch.is_tensor(v) and v.ndim >= 2) or "orig" in k:
            continue
        x = v.to(torch.float32)
        x = _lanczos_resize_chw(x, (Ht, Wt))
        if v.dtype == torch.bool:
            x = x > 0.5
        elif not torch.is_floating_point(v):
            x = x.round().to(v.dtype)
        sample[k] = x

    return sample


def run_inference(
    vae: AutoencoderKLQwenImage,
    transformer: QwenImageTransformer2DModel,
    embeds_dict: dict[str, torch.Tensor],
    processing_resolution: int,
    image_dir: str,
    output_dir: str,
    use_short_edge_tile=True,
    batch_size=2,
    num_workers=0,
):
    dataset = TilingDataset(
        transform_graph=functools.partial(
            data_transform, processing_resolution=processing_resolution
        ),
        input_folder=image_dir,
        gt_folder=image_dir,
        use_short_edge_tile=use_short_edge_tile,
        tiling_w=processing_resolution,
        tiling_h=processing_resolution,
        processing_resolution=processing_resolution,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    os.makedirs(output_dir, exist_ok=True)

    validate_single_dataset(
        vae,
        transformer,
        embeds_dict,
        data_loader=data_loader,
        save_to_dir=output_dir,
    )
