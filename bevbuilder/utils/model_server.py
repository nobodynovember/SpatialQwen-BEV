"""
Persistent model-server process.

Responsibilities:
  - Load CLIP (for text features) and LSeg (for per-pixel features) ONCE at startup.
  - Expose two kinds of requests via multiprocessing Queues:
      {"type": "init_info"}  → return text_feats, labels, colors, clip_feat_dim, norm_mean, norm_std
      {"type": "infer", "rgb": np.ndarray}  → return pix_feats np.ndarray
  - Quit cleanly when the sentinel value None is placed on the request queue.

Lifetime: one constant process managed by WorkerManager in bev_process.py.
"""

import math
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from bevbuilder.lseg.additional_utils.models import crop_image, pad_image, resize_image
from bevbuilder.lseg.modules.models.lseg_net import LSegEncNet

CROP_SIZE = 480
BASE_SIZE = 520


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ts_log(msg, log_path="bev_profile.log"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [model_server pid={os.getpid()}] {msg}"
    print(line, flush=True)
    try:
        with open(log_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_models():
    """Load CLIP text features and LSeg image encoder. Returns all state needed
    for inference and for initialising sim-worker instances."""
    df = pd.read_csv("bevbuilder/mpcat40.tsv", sep="\t")
    df = df[(df["mpcat40index"] >= 1) & (df["mpcat40index"] <= 40)]
    labels = df["mpcat40"].tolist()
    colors = [[int(h[1:][i: i + 2], 16) for i in (0, 2, 4)] for h in df["hex"]]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024, "RN101": 512, "RN50x4": 640, "RN50x16": 768,
        "RN50x64": 1024, "ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768,
    }[clip_version]

    _ts_log("loading CLIP model for text features …")
    clip_model, _ = clip.load(clip_version)
    clip_model.to(device).eval()
    lang_token = clip.tokenize(labels).to(device)
    with torch.no_grad():
        text_feats = clip_model.encode_text(lang_token)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    text_feats = text_feats.detach().cpu().numpy().copy().astype(np.float32)
    # Free CLIP weights – text features are pre-computed and stored as numpy
    del clip_model
    torch.cuda.empty_cache()

    _ts_log("loading LSeg model …")
    model = LSegEncNet(labels, arch_option=0, block_depth=0,
                       activation="lrelu", crop_size=CROP_SIZE)
    pretrained = torch.load("bevbuilder/lseg/checkpoints/demo_e200.ckpt", weights_only=False)
    pretrained = {k.lstrip("net."): v for k, v in pretrained["state_dict"].items()}
    model.load_state_dict(pretrained)
    model.eval().cuda()

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    return model, text_feats, transform, clip_feat_dim, labels, colors, norm_mean, norm_std


def _run_lseg_inference(model, rgb: np.ndarray, labels, transform, norm_mean, norm_std,
                        crop_size: int = CROP_SIZE, base_size: int = BASE_SIZE) -> np.ndarray:
    """Run LSeg on a single RGB image. Returns pix_feats (1, D, H, W) float32."""
    image = transform(rgb).unsqueeze(0).cuda()
    batch, _, h, w = image.size()
    stride = int(crop_size * (2.0 / 3.0))

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    cur_img = resize_image(image, height, width,
                           **{"mode": "bilinear", "align_corners": True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        with torch.no_grad():
            outputs, _ = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            pad_img = pad_image(cur_img, norm_mean, norm_std, crop_size)
        else:
            pad_img = cur_img
        _, _, ph, pw = pad_img.shape
        h_grids = int(math.ceil(1.0 * (ph - crop_size) / stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw - crop_size) / stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c, ph, pw).zero_().cuda()
                count_norm = image.new().resize_(batch, 1, ph, pw).zero_().cuda()
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                pad_crop_img = pad_image(crop_img, norm_mean, norm_std, crop_size)
                with torch.no_grad():
                    output, _ = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                outputs[:, :, h0:h1, w0:w1] += cropped
                count_norm[:, :, h0:h1, w0:w1] += 1
        assert (count_norm == 0).sum() == 0
        outputs = outputs / count_norm
        outputs = outputs[:, :, :height, :width]

    return outputs.cpu().numpy().copy().astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry-point: the server loop (run in a dedicated subprocess)
# ---------------------------------------------------------------------------

def model_server_loop(req_q, resp_q, ready_event, log_path="bev_profile.log"):
    """Main loop executed by the model-server subprocess.

    Parameters
    ----------
    req_q : multiprocessing.Queue
        Incoming requests from sim-worker subprocesses.
    resp_q : multiprocessing.Queue
        Outbound responses back to sim-worker subprocesses.
    ready_event : multiprocessing.Event
        Set once models are loaded and the server is ready to accept requests.
    log_path : str
        Path to the shared profile log file.
    """

    def _log(msg):
        _ts_log(msg, log_path=log_path)

    _log("starting … loading CLIP + LSeg models")
    t0 = time.perf_counter()
    model, text_feats, transform, clip_feat_dim, labels, colors, norm_mean, norm_std = _load_models()
    _log(f"models loaded dt={time.perf_counter() - t0:.3f}s labels={len(labels)} "
         f"clip_feat_dim={clip_feat_dim}")

    ready_event.set()

    req_count = 0
    while True:
        req = req_q.get()

        # Sentinel – clean shutdown
        if req is None:
            _log(f"received quit sentinel after {req_count} requests; exiting")
            break

        req_type = req.get("type")

        if req_type == "init_info":
            # A newly spawned sim-worker asks for the pre-computed tensors it
            # needs locally (text_feats for cosine similarity, label metadata …)
            resp_q.put({
                "ok": True,
                "text_feats": text_feats,
                "labels": labels,
                "colors": colors,
                "clip_feat_dim": clip_feat_dim,
                "norm_mean": norm_mean,
                "norm_std": norm_std,
            })
            req_count += 1
            continue

        if req_type == "infer":
            rgb = req["rgb"]
            try:
                pix_feats = _run_lseg_inference(
                    model, rgb, labels, transform, norm_mean, norm_std)
                resp_q.put({"ok": True, "pix_feats": pix_feats})
            except Exception as exc:
                import traceback
                resp_q.put({"ok": False, "err": str(exc),
                            "tb": traceback.format_exc()})
            req_count += 1
            continue

        # Unknown request type
        resp_q.put({"ok": False, "err": f"unknown request type: {req_type!r}"})
