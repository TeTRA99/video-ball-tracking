"""Overlay rendering functions for ball highlighting.

Every overlay has signature:
    overlay(frame_bgr, mask, *, frame_idx=0, history=None, **kwargs) -> frame_bgr

`mask` is HxW binary {0,1}. `frame_idx` lets overlays animate over time
(pulse, fades). `history` is a deque of recent (cx, cy) centroids for
trail-style overlays. Overlays that don't need those simply ignore them
via **kwargs.

All overlays are pure NumPy/OpenCV — no GPU required, runs in real time
at 1080p on a CPU.
"""
from __future__ import annotations

import cv2
import numpy as np


# --- shared helpers ---------------------------------------------------------

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


def _clamp_radius(raw: int, max_radius: int) -> int:
    """Cap drawing radius so a runaway big detection can't paint over
    half the frame. Real soccer ball in broadcast wide shot is ~10-20 px;
    even a generous max of 60 px lets us draw a clearly visible ring."""
    return min(raw, max_radius)


# --- static overlays --------------------------------------------------------

def ring(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 3,
    padding: int = 4,
    max_radius: int = 30,
    **_,
) -> np.ndarray:
    """Bright yellow ring around the ball, with a hard radius cap."""
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    _, _, w, h = bb
    radius = _clamp_radius(max(w, h) // 2 + padding, max_radius)
    out = frame.copy()
    cv2.circle(out, c, radius, color, thickness, lineType=cv2.LINE_AA)
    return out


def halo(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    blur: int = 31,
    alpha: float = 0.8,
    max_radius: int = 28,
    **_,
) -> np.ndarray:
    """Soft glowing halo around the ball (alpha-blended Gaussian)."""
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    H, W = frame.shape[:2]
    _, _, w, h = bb
    radius = _clamp_radius(max(w, h) // 2 + 10, max_radius)
    disk = np.zeros((H, W), dtype=np.float32)
    cv2.circle(disk, c, radius, 1.0, -1, lineType=cv2.LINE_AA)
    disk = cv2.GaussianBlur(disk, (blur, blur), 0)
    disk = np.clip(disk * alpha, 0, 1)[:, :, None]
    color_layer = np.zeros_like(frame, dtype=np.float32)
    color_layer[:] = color
    return (frame.astype(np.float32) * (1 - disk) + color_layer * disk).astype(np.uint8)


def arrow(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 0, 255),
    length: int = 60,
    max_offset: int = 80,
    **_,
) -> np.ndarray:
    """Downward arrow floating above the ball."""
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    cx, cy = c
    _, _, _, h = bb
    offset = min(h // 2 + 8, max_offset)
    tip = (cx, cy - offset)
    tail = (cx, tip[1] - length)
    out = frame.copy()
    cv2.arrowedLine(out, tail, tip, color, 4, line_type=cv2.LINE_AA, tipLength=0.4)
    return out


def recolor(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 255),
    **_,
) -> np.ndarray:
    """Repaint mask pixels with a high-contrast color."""
    if not mask.any():
        return frame
    out = frame.copy()
    out[mask > 0] = color
    return out


# --- animated overlays ------------------------------------------------------

def pulse(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    period: int = 16,
    max_radius: int = 45,
    frame_idx: int = 0,
    **_,
) -> np.ndarray:
    """Heartbeat ring: grows + fades each cycle of `period` frames.

    Effect: every `period` frames a ring expands outward from the ball and
    fades to transparent. Combined with high-contrast color, very easy to
    notice for low-vision viewers.
    """
    c = _centroid(mask)
    bb = _bbox(mask)
    if c is None or bb is None:
        return frame
    phase = (frame_idx % period) / period  # 0..1
    _, _, w, h = bb
    base_r = max(w, h) // 2 + 6
    radius = _clamp_radius(int(base_r + (max_radius - base_r) * phase), max_radius)
    # Fade out as the ring grows
    alpha = max(0.0, 1.0 - phase)
    overlay_layer = frame.copy()
    cv2.circle(overlay_layer, c, radius, color, 3, lineType=cv2.LINE_AA)
    # Also draw a tighter solid ring at the actual ball position for stability
    inner_r = _clamp_radius(base_r, max_radius)
    cv2.circle(overlay_layer, c, inner_r, color, 2, lineType=cv2.LINE_AA)
    return cv2.addWeighted(frame, 1 - 0.9 * alpha, overlay_layer, 0.9 * alpha, 0)


def spotlight(
    frame: np.ndarray,
    mask: np.ndarray,
    radius: int = 90,
    dim: float = 0.30,
    blur: int = 81,
    **_,
) -> np.ndarray:
    """Darken everything except a soft circle around the ball.

    Effect: scene becomes ~70% darker except the ball area, drawing the
    eye like a stage spotlight.
    """
    c = _centroid(mask)
    if c is None:
        return frame
    H, W = frame.shape[:2]
    disk = np.zeros((H, W), dtype=np.float32)
    cv2.circle(disk, c, radius, 1.0, -1, lineType=cv2.LINE_AA)
    disk = cv2.GaussianBlur(disk, (blur, blur), 0)
    disk = disk[:, :, None]
    dimmed = (frame.astype(np.float32) * dim)
    return (disk * frame.astype(np.float32) + (1 - disk) * dimmed).astype(np.uint8)


def trail(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    max_thickness: int = 5,
    history=None,
    **_,
) -> np.ndarray:
    """Fading trajectory line connecting recent ball positions."""
    out = frame.copy()
    pts = list(history) if history else []
    if len(pts) >= 2:
        n = len(pts)
        for i in range(1, n):
            alpha = i / n  # newer points fully opaque, older fade
            # Per-segment alpha-blend via a temp layer
            seg = frame.copy()
            thickness = max(1, int(max_thickness * alpha))
            cv2.line(seg, pts[i - 1], pts[i], color, thickness, lineType=cv2.LINE_AA)
            out = cv2.addWeighted(out, 1 - alpha * 0.7, seg, alpha * 0.7, 0)
    # Always draw a small marker at the current ball position
    c = _centroid(mask)
    if c is not None:
        cv2.circle(out, c, 6, color, 2, lineType=cv2.LINE_AA)
    return out


def chevron(
    frame: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    arm_length: int = 30,
    gap: int = 14,
    thickness: int = 3,
    **_,
) -> np.ndarray:
    """Broadcast-style chevron marks bracketing the ball (top + bottom).

    Two pairs of angled lines forming an open frame around the ball.
    Doesn't occlude the ball itself; reads as a graphics overlay.
    """
    c = _centroid(mask)
    if c is None:
        return frame
    cx, cy = c
    out = frame.copy()
    # Top chevron pointing down at the ball
    cv2.line(out, (cx - arm_length, cy - gap - arm_length),
             (cx, cy - gap), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(out, (cx + arm_length, cy - gap - arm_length),
             (cx, cy - gap), color, thickness, lineType=cv2.LINE_AA)
    # Bottom chevron pointing up at the ball
    cv2.line(out, (cx - arm_length, cy + gap + arm_length),
             (cx, cy + gap), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(out, (cx + arm_length, cy + gap + arm_length),
             (cx, cy + gap), color, thickness, lineType=cv2.LINE_AA)
    return out


OVERLAYS = {
    "ring": ring,
    "halo": halo,
    "arrow": arrow,
    "recolor": recolor,
    "pulse": pulse,
    "spotlight": spotlight,
    "trail": trail,
    "chevron": chevron,
}
