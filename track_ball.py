"""Highlight the ball in a soccer match video using YOLO26-seg.

Phase 1 MVP — pre-recorded MP4 in, annotated MP4 out.

Why YOLO26 instead of SAM 3.1: we hit hard limits running SAM 3.1's
multiplex video predictor on 8 GB of consumer VRAM (model assumes H100-
class hardware, hardcodes FP8 attention, OOMs at 540p). YOLO26-seg:
- Fits easily in 8 GB VRAM
- ~30+ FPS on RTX 4070 Laptop
- COCO 'sports ball' class (id 32) matches soccer balls in the
  pretrained model with no fine-tuning
- Better small-object detection than YOLO11 — crucial for tracking
  a ball that's far from the camera
- Returns per-instance binary masks (.masks.data), which feeds straight
  into our existing overlays.py via a small union step

The original SAM 3.1 implementation is preserved at track_ball_sam3.py
for when we have access to bigger GPUs (or as a refinement pass on
selected key frames later).
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Iterator

import click
import cv2
import numpy as np
from tqdm import tqdm

from overlays import OVERLAYS


# COCO class id 32 = "sports ball". The pretrained YOLO26 model uses
# the COCO label set; we filter to this class to avoid drawing rings on
# people, cleats, the ref's whistle, etc.
SPORTS_BALL_CLASS_ID = 32


def build_predictor(model_size: str = "n"):
    """Construct a YOLO26-seg model.

    model_size in {n, s, m, l, x}. n (nano) is plenty for ball detection
    on a 4070 Laptop and downloads in seconds. Bump up if accuracy on
    small/distant balls is poor.
    """
    from ultralytics import YOLO
    return YOLO(f"yolo26{model_size}-seg.pt")


def _select_best_mask(
    result,
    frame_hw: tuple[int, int],
    max_ball_px: int = 35,
    max_aspect: float = 1.5,
) -> np.ndarray | None:
    """Pick the SINGLE highest-confidence detection that passes filters and
    return its mask. Returns None if no detection qualifies.

    Picking one (instead of unioning all) eliminates per-frame jumpiness
    when YOLO finds multiple "balls" — typically the real ball plus a few
    false positives on heads/shadows. Picking the most confident one is
    a reliable heuristic at this scale.

    Filters:
    - longest bbox edge in [4, max_ball_px] (real ball in 1080p is ~15-25
      px; anything > 35 is almost always a head/shadow/banner)
    - aspect ratio <= max_aspect (ball is roughly square)
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None
    H, W = frame_hw

    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else [1.0] * len(result.boxes)
    candidates: list[tuple[float, int]] = []
    for i, box in enumerate(result.boxes.xywh):
        _, _, w, h = box.tolist()
        if not (4 <= max(w, h) <= max_ball_px):
            continue
        if min(w, h) > 0 and max(w, h) / min(w, h) > max_aspect:
            continue
        candidates.append((confs[i], i))

    if not candidates:
        return None

    _, best_i = max(candidates)
    m = result.masks.data[best_i]
    bm = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    if bm.shape != (H, W):
        bm = cv2.resize(bm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return (bm > 0.5).astype(np.uint8)


def iter_predictions(
    predictor,
    video_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    max_ball_px: int = 35,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Yield (frame_bgr, mask_or_None) per frame.

    Ultralytics' stream=True returns one Result per frame, in order.
    OpenCV reads the same video alongside so we get the raw BGR pixels
    for overlay rendering.

    imgsz is the long-edge size YOLO resizes the input to. The default
    640 is too aggressive for broadcast wide shots where the ball is
    ~10-15 px in source — feeding higher imgsz preserves the pixels
    YOLO needs to recognize the ball. 1280 is a good first bump;
    1920 keeps 1080p at full detail (but uses ~4x more VRAM than 640).
    """
    results = predictor(
        source=str(video_path),
        stream=True,
        classes=[SPORTS_BALL_CLASS_ID],
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )
    cap = cv2.VideoCapture(str(video_path))
    try:
        for result in results:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            mask = _select_best_mask(result, frame_bgr.shape[:2], max_ball_px=max_ball_px)
            yield frame_bgr, mask
    finally:
        cap.release()


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input video file (MP4).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output annotated MP4 path.",
)
@click.option(
    "--overlay",
    type=click.Choice(list(OVERLAYS)),
    default="ring",
    show_default=True,
    help="Overlay style to draw on top of the ball.",
)
@click.option(
    "--model-size",
    type=click.Choice(["n", "s", "m", "l", "x"]),
    default="n",
    show_default=True,
    help="YOLO26-seg model size. n is fastest, x is most accurate.",
)
@click.option(
    "--conf",
    type=float,
    default=0.25,
    show_default=True,
    help="Detection confidence threshold (lower catches more, risks false positives).",
)
@click.option(
    "--imgsz",
    type=int,
    default=640,
    show_default=True,
    help="YOLO long-edge resize. Bump to 1280 or 1920 for broadcast wide shots "
         "where the ball is tiny in the source frame.",
)
@click.option(
    "--max-ball-px",
    type=int,
    default=35,
    show_default=True,
    help="Reject 'ball' detections larger than this on the longest edge. "
         "Real soccer ball in 1080p wide shot is ~15-25 px; over 35 is "
         "almost always a head/shadow/banner false positive.",
)
@click.option(
    "--max-jump-px",
    type=int,
    default=150,
    show_default=True,
    help="If the ball appears to move more than this many pixels between "
         "consecutive hit frames, treat as a false positive and skip drawing. "
         "Real ball can't teleport across the frame in 1/25 of a second.",
)
@click.option(
    "--trail-frames",
    type=int,
    default=30,
    show_default=True,
    help="How many recent ball positions to remember for the 'trail' overlay.",
)
def main(
    input_path: Path,
    output_path: Path,
    overlay: str,
    model_size: str,
    conf: float,
    imgsz: int,
    max_ball_px: int,
    max_jump_px: int,
    trail_frames: int,
) -> None:
    """Annotate a soccer video by overlaying a graphic on the ball."""
    overlay_fn = OVERLAYS[overlay]

    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    click.echo(f"Loading yolo26{model_size}-seg…")
    predictor = build_predictor(model_size)
    click.echo(
        f"Tracking 'sports ball'  overlay: {overlay}  conf>={conf}  "
        f"imgsz={imgsz}  max_ball_px={max_ball_px}  max_jump_px={max_jump_px}  "
        f"frames: {n_frames}"
    )

    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    t0 = time.perf_counter()
    n_done = 0
    n_hits = 0
    n_jumps_rejected = 0
    prev_centroid: tuple[int, int] | None = None
    frames_since_hit = 0
    for frame_idx, (frame, mask) in enumerate(tqdm(
        iter_predictions(
            predictor, input_path,
            conf=conf, imgsz=imgsz, max_ball_px=max_ball_px,
        ),
        total=n_frames,
        desc=overlay,
    )):
        if mask is not None and mask.any():
            ys, xs = np.where(mask > 0)
            cx, cy = int(xs.mean()), int(ys.mean())

            # Jump filter scales with the gap since the last hit. If YOLO
            # missed N frames between detections, a real ball could have
            # moved ~max_jump_px*N pixels legitimately. Without this
            # scaling, prev_centroid gets stuck at a stale position when
            # the ball moves across the field while YOLO is silent.
            if prev_centroid is not None:
                allowed = max_jump_px * (1 + frames_since_hit)
                dx = cx - prev_centroid[0]
                dy = cy - prev_centroid[1]
                if (dx * dx + dy * dy) > (allowed * allowed):
                    n_jumps_rejected += 1
                    # Don't update prev_centroid — wait for a detection
                    # closer to it. But DO count this as a missed frame so
                    # the allowance grows.
                    frames_since_hit += 1
                    writer.write(frame)
                    n_done += 1
                    continue

            history.append((cx, cy))
            prev_centroid = (cx, cy)
            frames_since_hit = 0
            frame = overlay_fn(frame, mask, frame_idx=frame_idx, history=history)
            n_hits += 1
        else:
            frames_since_hit += 1
        writer.write(frame)
        n_done += 1

    writer.release()
    elapsed = time.perf_counter() - t0
    fps_actual = n_done / elapsed if elapsed > 0 else 0.0
    click.echo(
        f"Wrote {output_path}\n"
        f"  frames={n_done} hits={n_hits} ({100*n_hits/max(n_done,1):.1f}%) "
        f"jumps_rejected={n_jumps_rejected} "
        f"elapsed={elapsed:.1f}s avg_fps={fps_actual:.2f}"
    )


if __name__ == "__main__":
    main()
