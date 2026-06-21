"""
Attention heatmap generation.

Converts per-step cross-attention weights → a coloured overlay on the input image.
"""

from __future__ import annotations

import base64
import io
import numpy as np
from PIL import Image
import cv2


TYPE_COLORS = {
    "text":   (59,  130, 246), 
    "visual": (34,  197, 94), 
    "latent": (249, 115, 22), 
}

COLORMAP = cv2.COLORMAP_JET


def attn_to_heatmap_overlay(
    image: Image.Image,
    attn_weights: np.ndarray | None,
    grid_hw: tuple[int, int],
    alpha: float = 0.5,
    image_token_range: tuple[int, int] | None = None,
) -> str:
    """
    Convert cross-attention weights for one step into a heatmap overlaid on image.

    Parameters
    ----------
    image        : original PIL Image
    attn_weights : (num_heads, src_len) attention weights, or None
    grid_hw      : (grid_h, grid_w) spatial grid of image patches
    alpha        : blending weight for overlay (0=image only, 1=heatmap only)
    image_token_range : (start, end) indices of image tokens in the input sequence
    Returns
    -------
    base64-encoded PNG string (for use in html.Img src)
    """
    img_np = np.array(image.convert("RGB"))
    h_img, w_img = img_np.shape[:2]

    if attn_weights is None:
        return _encode_image(img_np)

    grid_h, grid_w = grid_hw

    num_img_tokens = grid_h * grid_w

    # attn_weights may be (num_heads, kv_len) raw or (kv_len,) already head-averaged
    mean_attn = attn_weights.mean(axis=0) if attn_weights.ndim == 2 else attn_weights

    if image_token_range is not None:
        start, end = image_token_range
        img_attn = mean_attn[start:end]
    else:
        img_attn = mean_attn[:num_img_tokens]

    target_len = grid_h * grid_w
    if img_attn.shape[0] < target_len:
        img_attn = np.pad(img_attn, (0, target_len - img_attn.shape[0]))
    else:
        img_attn = img_attn[:target_len]

    attn_grid = img_attn.reshape(grid_h, grid_w).astype(np.float32)
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)

    attn_resized = cv2.resize(attn_grid, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    attn_uint8 = (attn_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(attn_uint8, COLORMAP)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * img_np).astype(np.uint8)
    return _encode_image(overlay)


def _encode_image(img_np: np.ndarray) -> str:
    """Encode a numpy RGB array as a base64 PNG data URI."""
    pil = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def encode_pil_image(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URI (no overlay)."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
