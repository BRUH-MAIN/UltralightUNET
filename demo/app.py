"""Interactive demo: upload a skin-lesion image (or pick a test-set sample),
pick a dataset's trained model, and see the segmentation live.

Ties together every earlier week's work rather than reimplementing any of it:
  - checkpoint lookup / GPU model loading:   scripts/cross_eval.py
  - CPU model + INT8 dynamic quantization:   scripts/quantize_eval.py
  - spatial attention + Seg-Grad-CAM:        scripts/explainability.py

Two toggles: "full precision" (GPU if available, else CPU fp32) vs "INT8
quantized (CPU)" -- shows the Week 4 size/latency difference live, measured
on this machine rather than quoted from COMPARISON.md -- and "show attention
/ Grad-CAM", which only works in full-precision mode: quantized Linear
layers are inference-only (no backward()), so Seg-Grad-CAM can't run against
them. That combination is disabled in the UI rather than silently ignored.

Usage:
    python demo/app.py
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import gradio as gr
import numpy as np
import torch
from PIL import Image

from configs.config_setting import setting_config
from scripts.explainability import compute_maps
from scripts.cross_eval import DATASETS, find_best_checkpoint, load_model as load_gpu_model
from loader import isic_loader
from scripts.quantize_eval import build_cpu_model, load_weights as load_cpu_weights, state_dict_size_mb

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_dataset_cache = {}   # dataset -> isic_loader test split
_model_cache = {}     # (dataset, mode) -> (model, device, size_mb, n_params)


def _test_dataset(dataset):
    if dataset not in _dataset_cache:
        data_path = os.path.join(REPO_ROOT, 'data', dataset) + os.sep
        _dataset_cache[dataset] = isic_loader(path_Data=data_path, train=False, Test=True)
    return _dataset_cache[dataset]


def _get_model(dataset, mode):
    """mode: 'full' (GPU if available, else CPU fp32) or 'int8' (CPU, quantized)."""
    key = (dataset, mode)
    if key in _model_cache:
        return _model_cache[key]

    ckpt_path, _ = find_best_checkpoint(dataset)
    if mode == 'full' and DEVICE == 'cuda':
        try:
            model = load_gpu_model(ckpt_path, setting_config.model_config)
            device = 'cuda'
        except (torch.OutOfMemoryError, RuntimeError) as e:
            # GPU may be busy with other work (this box is not dedicated to this
            # demo) -- fall back to CPU fp32 rather than crash the app.
            print(f"GPU unavailable ({e}); falling back to CPU for '{dataset}' full-precision model")
            torch.cuda.empty_cache()
            model = build_cpu_model(setting_config.model_config)
            load_cpu_weights(model, ckpt_path)
            device = 'cpu'
    else:
        model = build_cpu_model(setting_config.model_config)
        load_cpu_weights(model, ckpt_path)
        device = 'cpu'
        if mode == 'int8':
            model = torch.quantization.quantize_dynamic(model, {
                name for name, mod in model.named_modules()
                if isinstance(mod, torch.nn.Linear) and '.mamba.' not in name
            }, dtype=torch.qint8)

    n_params = sum(p.numel() for p in model.parameters()) if mode != 'int8' else None
    size_mb = state_dict_size_mb(model)
    result = (model, device, size_mb, n_params)
    _model_cache[key] = result
    return result


def load_sample(dataset, idx):
    ds = _test_dataset(dataset)
    idx = int(idx) % len(ds)
    img_t, msk_t = ds[idx]
    disp = img_t.permute(1, 2, 0).numpy()
    disp = ((disp - disp.min()) / (disp.max() - disp.min() + 1e-8) * 255).astype(np.uint8)
    mask = (msk_t.squeeze().numpy() >= 0.5).astype(np.uint8) * 255
    return disp, mask, f"loaded {dataset} test image #{idx}"


def _preprocess(image_np):
    """Match dataprepare/Prepare_*.py's pipeline: 256x256 bilinear resize, then
    whole-image min-max stretch to [0,255]. dataset_normalized's global
    mean/std step is an affine transform that a following min-max normalize
    cancels out exactly, so this is the correct per-image equivalent for a
    single uploaded image with no dataset statistics to draw on."""
    img = Image.fromarray(image_np).convert('RGB').resize((256, 256), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    t = torch.tensor(arr).permute(2, 0, 1)
    return t


def dice(pred, gt, threshold=0.5):
    p = (pred >= threshold).astype(np.float32)
    g = (gt >= 0.5).astype(np.float32)
    inter = (p * g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / denom) if denom else None


def overlay_mask(image_uint8, mask01, color=(255, 60, 60), alpha=0.4):
    out = image_uint8.astype(np.float32).copy()
    m = mask01.astype(bool)
    for c in range(3):
        out[..., c][m] = out[..., c][m] * (1 - alpha) + color[c] * alpha
    return out.astype(np.uint8)


def run_inference(image_np, dataset, gt_mask, mode, show_explain):
    if image_np is None:
        return None, None, None, "Upload an image or load a sample first."
    if mode == 'int8' and show_explain:
        show_explain = False
        note_prefix = "(attention/Grad-CAM unavailable for the INT8 model -- quantized " \
                      "Linear layers don't support backward(); showing prediction only)\n\n"
    else:
        note_prefix = ""

    img_t = _preprocess(image_np)
    model, device, size_mb, n_params = _get_model(dataset, mode)

    if show_explain:
        img = img_t.unsqueeze(0).to(device).float()
        img.requires_grad_(True)
        t0 = time.perf_counter()
        prob, satt, cam = compute_maps(model, img)
        latency_ms = (time.perf_counter() - t0) * 1000
    else:
        img = img_t.unsqueeze(0).to(device).float()
        with torch.no_grad():
            t0 = time.perf_counter()
            prob = model(img).squeeze().cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000
        satt = cam = None

    disp = img_t.permute(1, 2, 0).numpy().astype(np.uint8)
    pred_mask = (prob >= 0.5).astype(np.uint8)
    pred_overlay = overlay_mask(disp, pred_mask, color=(255, 60, 60))

    info = note_prefix
    info += f"model: {dataset}  |  mode: {'INT8 quantized (CPU)' if mode == 'int8' else f'full precision ({device})'}\n"
    info += f"state_dict size: {size_mb:.3f} MB"
    if n_params is not None:
        info += f"  ({n_params:,} params)"
    info += f"\ninference latency: {latency_ms:.1f} ms (this call, single image, {device})\n"

    if gt_mask is not None:
        gt01 = (np.array(gt_mask) >= 128).astype(np.float32)
        d = dice(prob, gt01)
        info += f"DSC vs. ground truth: {d:.4f}\n"

    attn_img = None
    cam_img = None
    if satt is not None:
        attn_img = overlay_mask(disp, (satt >= np.percentile(satt, 80)).astype(np.uint8), color=(80, 120, 255), alpha=0.5)
    if cam is not None:
        cam_img = overlay_mask(disp, (cam >= np.percentile(cam, 80)).astype(np.uint8), color=(255, 170, 0), alpha=0.5)

    return pred_overlay, attn_img, cam_img, info


with gr.Blocks(title="UltraLight VM-UNet demo") as demo:
    gr.Markdown(
        "# UltraLight VM-UNet -- skin lesion segmentation\n"
        "49,457-parameter model (Wu et al., *Patterns* 2025, replicated in this repo). "
        "Pick a dataset's trained model, load a test-set sample or upload your own dermoscopy "
        "image, and run it. See `results/COMPARISON.md` for the full replication and analysis."
    )

    gt_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            dataset_dd = gr.Dropdown(DATASETS, value='ISIC2018', label="Dataset (selects the trained model)")
            with gr.Row():
                sample_idx = gr.Number(value=42, precision=0, label="Test-set sample index")
                load_btn = gr.Button("Load sample")
            image_input = gr.Image(label="Input image (upload or load a sample above)", type='numpy')
            mode_radio = gr.Radio(['full', 'int8'], value='full',
                                  label="Model", info="full = full precision (GPU if available); "
                                                       "int8 = INT8 dynamic-quantized (CPU)")
            explain_check = gr.Checkbox(value=True, label="Show spatial attention + Seg-Grad-CAM "
                                                          "(full precision only)")
            run_btn = gr.Button("Run", variant='primary')
        with gr.Column():
            pred_out = gr.Image(label="Prediction (overlay)")
            with gr.Row():
                attn_out = gr.Image(label="Spatial attention (top 20%, overlay)")
                cam_out = gr.Image(label="Seg-Grad-CAM (top 20%, overlay)")
            info_out = gr.Textbox(label="Info", lines=5)

    load_btn.click(load_sample, inputs=[dataset_dd, sample_idx],
                   outputs=[image_input, gt_state, info_out])
    image_input.upload(lambda: None, outputs=[gt_state])
    run_btn.click(run_inference, inputs=[image_input, dataset_dd, gt_state, mode_radio, explain_check],
                 outputs=[pred_out, attn_out, cam_out, info_out])


if __name__ == "__main__":
    demo.launch()
