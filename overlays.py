"""Overlay rendering functions for ball highlighting.

Each function takes (frame_bgr, binary_mask) and returns the annotated frame.
binary_mask is HxW with values {0, 1} (or {False, True}).

All four are pure NumPy/OpenCV — they can be unit-tested on synthetic masks
without a GPU or SAM 3 installed.
"""
from __future__ import annotations

import cv2
import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())


def _centroid(mask: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.mean()), int(ys.mean())


def ring(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 4,
    padding: int = 8,
) -> np.ndarray:
    """Bright ring around the ball's bounding circle."""
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    _, _, w, h = bb
    radius = max(w, h) // 2 + padding
    out = frame.copy()
    cv2.circle(out, c, radius, color, thickness, lineType=cv2.LINE_AA)
    return out


def halo(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    blur: int = 21,
    alpha: float = 0.7,
) -> np.ndarray:
    """Soft glowing halo around the ball."""
    if not mask.any():
        return frame
    m = (mask > 0).astype(np.uint8) * 255
    dilated = cv2.dilate(m, np.ones((15, 15), np.uint8))
    halo_mask = cv2.GaussianBlur(dilated, (blur, blur), 0).astype(np.float32) / 255.0
    halo_mask = np.clip(halo_mask * alpha, 0, 1)[:, :, None]

    color_layer = np.zeros_like(frame, dtype=np.float32)
    color_layer[:] = color
    out = frame.astype(np.float32) * (1 - halo_mask) + color_layer * halo_mask
    return out.astype(np.uint8)


def arrow(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    length: int = 60,
) -> np.ndarray:
    """Downward arrow floating above the ball."""
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    cx, cy = c
    _, _, _, h = bb
    tip = (cx, cy - h // 2 - 8)
    tail = (cx, tip[1] - length)
    out = frame.copy()
    cv2.arrowedLine(out, tail, tip, color, 4, line_type=cv2.LINE_AA, tipLength=0.4)
    return out


def recolor(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 255),
) -> np.ndarray:
    """Recolor the mask pixels with a high-contrast color."""
    if not mask.any():
        return frame
    out = frame.copy()
    out[mask > 0] = color
    return out


OVERLAYS = {
    "ring": ring,
    "halo": halo,
    "arrow": arrow,
    "recolor": recolor,
}
