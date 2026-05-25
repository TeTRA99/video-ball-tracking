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
from tracker import BallTracker


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
    help="Per-frame ball motion limit for the tracker. Allowance scales with "
         "the gap since the last detection.",
)
@click.option(
    "--max-extrapolate",
    type=int,
    default=8,
    show_default=True,
    help="When YOLO misses the ball, the tracker keeps drawing the overlay "
         "at the predicted (extrapolated) position for this many frames. "
         "After that, it gives up until the next real detection.",
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
    max_extrapolate: int,
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
        f"max_extrapolate={max_extrapolate}  frames: {n_frames}"
    )

    tracker = BallTracker(
        max_jump_per_frame=max_jump_px,
        max_extrapolate_frames=max_extrapolate,
    )
    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    # Cache the most recent real mask so we can re-draw it (translated to
    # the predicted position) during extrapolation frames. Avoids having
    # to synthesize a disk when we already know the ball's shape.
    last_real_mask: np.ndarray | None = None
    last_real_centroid: tuple[int, int] | None = None

    t0 = time.perf_counter()
    n_done = 0
    n_hits = 0       # real detections drawn
    n_predicted = 0  # extrapolated frames drawn
    for frame_idx, (frame, mask) in enumerate(tqdm(
        iter_predictions(
            predictor, input_path,
            conf=conf, imgsz=imgsz, max_ball_px=max_ball_px,
        ),
        total=n_frames,
        desc=overlay,
    )):
        # Convert detection -> centroid (or None)
        if mask is not None and mask.any():
            ys, xs = np.where(mask > 0)
            detection = (int(xs.mean()), int(ys.mean()))
        else:
            detection = None

        result = tracker.feed(detection)
        if result is not None:
            (cx, cy), is_real = result
            history.append((cx, cy))

            if is_real:
                draw_mask = mask
                last_real_mask = mask
                last_real_centroid = (cx, cy)
                n_hits += 1
            else:
                # Extrapolated frame: shift the last real mask to the
                # predicted centroid so all overlay shapes work unchanged.
                if last_real_mask is not None and last_real_centroid is not None:
                    dx = cx - last_real_centroid[0]
                    dy = cy - last_real_centroid[1]
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    draw_mask = cv2.warpAffine(
                        last_real_mask, M, (last_real_mask.shape[1], last_real_mask.shape[0]),
                        flags=cv2.INTER_NEAREST, borderValue=0,
                    )
                else:
                    draw_mask = None
                n_predicted += 1

            if draw_mask is not None:
                frame = overlay_fn(frame, draw_mask, frame_idx=frame_idx, history=history)

        writer.write(frame)
        n_done += 1

    writer.release()
    elapsed = time.perf_counter() - t0
    fps_actual = n_done / elapsed if elapsed > 0 else 0.0
    total_drawn = n_hits + n_predicted
    click.echo(
        f"Wrote {output_path}\n"
        f"  frames={n_done} hits={n_hits} predicted={n_predicted} "
        f"drawn={total_drawn} ({100*total_drawn/max(n_done,1):.1f}%) "
        f"elapsed={elapsed:.1f}s avg_fps={fps_actual:.2f}"
    )


if __name__ == "__main__":
    main()
