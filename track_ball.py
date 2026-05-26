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


# Default ball class id is 32 (COCO "sports ball" in the stock YOLO26 model).
# After fine-tuning on football-players-detection, ball is class 0 — pass
# --ball-class 0 in that case.
DEFAULT_BALL_CLASS_ID = 32


def build_predictor(model_size: str = "n", weights_path: str | None = None):
    """Construct a YOLO26 model.

    If weights_path is given, load those custom weights (e.g. fine-tuned
    `runs/ball_finetune_v1/weights/best.pt`). Otherwise load the stock
    pretrained yolo26{size}-seg.pt — the -seg variant gives binary masks
    that feed directly into overlays.

    Fine-tuned weights from the football-players-detection dataset are
    detection-only (no masks). The iter_predictions code below handles
    both cases by synthesizing a disk mask from the bbox when masks are
    absent.
    """
    from ultralytics import YOLO
    if weights_path:
        return YOLO(weights_path)
    return YOLO(f"yolo26{model_size}-seg.pt")


def _select_best_mask(
    result,
    frame_hw: tuple[int, int],
    max_ball_px: int = 35,
    max_aspect: float = 1.5,
    prev_centroid: tuple[int, int] | None = None,
    proximity_px: int = 200,
    sticky_id: int | None = None,
) -> tuple[np.ndarray | None, int | None, float]:
    """Pick a single detection's mask and return (mask, picked_tracker_id).

    Selection priority:
    1. If `sticky_id` is given AND a current candidate has that ByteTrack/
       BoT-SORT id, use that detection. ByteTrack maintains identity across
       frames via Kalman filtering, so once we've locked onto the ball's
       id we keep tracking it even when a player's head briefly outscores
       the ball in confidence.
    2. If `prev_centroid` is given AND multiple candidates pass filters,
       prefer candidates within `proximity_px` of the previous position.
    3. Otherwise pick the highest-confidence candidate.

    Filters before any selection: longest bbox edge in [4, max_ball_px],
    aspect ratio <= max_aspect (ball is roughly square).

    Returns (mask, picked_id) so callers can thread sticky_id forward.
    picked_id is None if the detector wasn't run in tracking mode.
    """
    if result.boxes is None or len(result.boxes) == 0:
        return None, None, 0.0
    H, W = frame_hw

    confs = result.boxes.conf.tolist() if result.boxes.conf is not None else [1.0] * len(result.boxes)
    # When predictor.track() is used, .id is a tensor of ByteTrack/BoT-SORT
    # ids. In plain predict mode, .id is None — we degrade gracefully.
    raw_ids = result.boxes.id
    if raw_ids is not None:
        ids_list = [int(i) for i in raw_ids.tolist()]
    else:
        ids_list = [None] * len(result.boxes)

    candidates: list[tuple[float, int, tuple[float, float, float, float], int | None]] = []
    for i, box in enumerate(result.boxes.xywh):
        cx, cy, w, h = box.tolist()
        if not (4 <= max(w, h) <= max_ball_px):
            continue
        if min(w, h) > 0 and max(w, h) / min(w, h) > max_aspect:
            continue
        candidates.append((confs[i], i, (cx, cy, w, h), ids_list[i]))

    if not candidates:
        return None, None, 0.0

    selected = None
    # 1. Prefer the locked id, if it survives the current frame's filters.
    if sticky_id is not None:
        for c in candidates:
            if c[3] == sticky_id:
                selected = c
                break
    # 2. Trajectory proximity as a tiebreaker among multiple candidates.
    if selected is None:
        pool = candidates
        if prev_centroid is not None and len(pool) > 1:
            px, py = prev_centroid
            nearby = [
                c for c in pool
                if (c[2][0] - px) ** 2 + (c[2][1] - py) ** 2 <= proximity_px ** 2
            ]
            if nearby:
                pool = nearby
        selected = max(pool)

    picked_conf, best_i, (cx, cy, w, h), picked_id = selected

    if result.masks is not None and best_i < len(result.masks.data):
        m = result.masks.data[best_i]
        bm = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
        if bm.shape != (H, W):
            bm = cv2.resize(bm.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        return (bm > 0.5).astype(np.uint8), picked_id, float(picked_conf)

    mask = np.zeros((H, W), dtype=np.uint8)
    radius = max(2, int(max(w, h) / 2))
    cv2.circle(mask, (int(cx), int(cy)), radius, 1, -1)
    return mask, picked_id, float(picked_conf)


def iter_predictions(
    predictor,
    video_path: Path,
    conf: float = 0.25,
    imgsz: int = 640,
    ball_class: int = DEFAULT_BALL_CLASS_ID,
    tracker: str = "bytetrack.yaml",
) -> Iterator[tuple[np.ndarray, object]]:
    """Yield (frame_bgr, raw_result) per frame. The caller picks the best
    mask from raw_result — this lets the main loop pass tracker state in.

    Uses Ultralytics' model.track() which runs ByteTrack (default) or
    BoT-SORT on top of YOLO and assigns a stable id to each detection
    across frames (Kalman-filtered motion association). result.boxes.id
    surfaces those ids; _select_best_mask uses them to lock the overlay
    onto one tracked instance instead of jumping each frame to whatever
    YOLO scored highest.

    ball_class is the model's class id for "ball": 32 for stock YOLO26
    (COCO 'sports ball'), 0 for our fine-tuned model. Set via --ball-class.
    """
    results = predictor.track(
        source=str(video_path),
        stream=True,
        classes=[ball_class],
        conf=conf,
        imgsz=imgsz,
        tracker=tracker,
        verbose=False,
        persist=False,  # new session per file source
    )
    cap = cv2.VideoCapture(str(video_path))
    try:
        for result in results:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            yield frame_bgr, result
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
    help="YOLO26-seg model size. n is fastest, x is most accurate. "
         "Ignored when --model is given.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a custom weights file (e.g. our fine-tuned "
         "runs/ball_finetune_v1/weights/best.pt). Overrides --model-size.",
)
@click.option(
    "--ball-class",
    type=int,
    default=DEFAULT_BALL_CLASS_ID,
    show_default=True,
    help="Class id for 'ball' in the model. 32 is COCO 'sports ball' "
         "(stock model); set to 0 for our fine-tuned model.",
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
    default=40,
    show_default=True,
    help="Reject 'ball' detections larger than this on the longest edge. "
         "Real soccer ball in 1080p wide shot is ~15-25 px; over ~40 is "
         "almost always a head/shadow/banner false positive. Bump to 60+ "
         "for close-up footage where the ball is genuinely larger.",
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
    default=3,
    show_default=True,
    help="When YOLO misses the ball, draw the overlay at the predicted "
         "(extrapolated) position for this many frames. Constant-velocity "
         "prediction drifts noticeably past ~3 frames in real soccer "
         "(direction changes accumulate), but 1-3 frames is invisible "
         "smoothing that bridges brief detection misses. Set to 0 to "
         "disable entirely; raise to 6-8 for slow/predictable footage.",
)
@click.option(
    "--trail-frames",
    type=int,
    default=30,
    show_default=True,
    help="How many recent ball positions to remember for the 'trail' overlay.",
)
@click.option(
    "--tracker",
    "tracker_yaml",
    type=str,
    default="bytetrack.yaml",
    show_default=True,
    help="Ultralytics tracker config. Built-in names: 'bytetrack.yaml' "
         "(Kalman motion, default) or 'botsort.yaml' (motion + ReID, "
         "heavier). Or pass a path to a custom yaml — we ship "
         "'trackers/bytetrack_ball.yaml' tuned for small-fast-object "
         "tracking with stickier ids during brief occlusions.",
)
def main(
    input_path: Path,
    output_path: Path,
    overlay: str,
    model_size: str,
    model_path: Path | None,
    ball_class: int,
    conf: float,
    imgsz: int,
    max_ball_px: int,
    max_jump_px: int,
    max_extrapolate: int,
    trail_frames: int,
    tracker_yaml: str,
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

    if model_path:
        click.echo(f"Loading custom weights: {model_path}")
    else:
        click.echo(f"Loading yolo26{model_size}-seg…")
    predictor = build_predictor(
        model_size,
        weights_path=str(model_path) if model_path else None,
    )
    click.echo(
        f"Tracking ball (class={ball_class})  overlay: {overlay}  conf>={conf}  "
        f"imgsz={imgsz}  max_ball_px={max_ball_px}  max_jump_px={max_jump_px}  "
        f"max_extrapolate={max_extrapolate}  frames: {n_frames}"
    )

    tracker = BallTracker(
        max_jump_per_frame=max_jump_px,
        max_extrapolate_frames=max_extrapolate,
    )
    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    last_real_mask: np.ndarray | None = None
    last_real_centroid: tuple[int, int] | None = None
    sticky_id: int | None = None
    # Hybrid overlay-alpha state:
    #   - jump_fade_remaining is decremented each frame after a teleport-
    #     sized jump; while > 0 we suppress the overlay so the viewer
    #     never sees the hard position change.
    #   - prev_draw_centroid is the last centroid we actually drew; we
    #     compare against it to detect jumps.
    prev_draw_centroid: tuple[int, int] | None = None
    jump_fade_remaining = 0
    # 15 frames ~= 500ms at 30fps. Linear ramp: invisible at the jump frame,
    # fully visible at frame +15. Long enough that the viewer's eye reads
    # the gap as "the overlay is reorienting" rather than a teleport.
    JUMP_FADE_FRAMES = 15
    JUMP_THRESHOLD_PX = 150

    t0 = time.perf_counter()
    n_done = 0
    n_hits = 0       # real detections drawn
    n_predicted = 0  # extrapolated frames drawn
    for frame_idx, (frame, raw_result) in enumerate(tqdm(
        iter_predictions(
            predictor, input_path,
            conf=conf, imgsz=imgsz,
            ball_class=ball_class,
            tracker=tracker_yaml,
        ),
        total=n_frames,
        desc=overlay,
    )):
        mask, picked_id, picked_conf = _select_best_mask(
            raw_result, frame.shape[:2],
            max_ball_px=max_ball_px,
            prev_centroid=tracker.last_centroid,
            sticky_id=sticky_id,
        )
        if sticky_id is None and picked_id is not None:
            sticky_id = picked_id

        if mask is not None and mask.any():
            ys, xs = np.where(mask > 0)
            detection = (int(xs.mean()), int(ys.mean()))
        else:
            detection = None

        tracked = tracker.feed(detection)
        if tracked is not None:
            (cx, cy), is_real = tracked
            history.append((cx, cy))

            if is_real:
                draw_mask = mask
                last_real_mask = mask
                last_real_centroid = (cx, cy)
                n_hits += 1
            else:
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
                # Jump detection: if the new draw centroid is implausibly
                # far from the previous one, trigger a brief fade so the
                # viewer sees a quick blink, not a hard teleport.
                if prev_draw_centroid is not None:
                    dx = cx - prev_draw_centroid[0]
                    dy = cy - prev_draw_centroid[1]
                    if dx * dx + dy * dy > JUMP_THRESHOLD_PX * JUMP_THRESHOLD_PX:
                        jump_fade_remaining = JUMP_FADE_FRAMES
                prev_draw_centroid = (cx, cy)

                # Only the jump fade modulates alpha now. Stable detections
                # draw at full opacity regardless of confidence.
                if jump_fade_remaining > 0:
                    alpha = 1.0 - (jump_fade_remaining / JUMP_FADE_FRAMES)
                    jump_fade_remaining -= 1
                else:
                    alpha = 1.0

                if alpha > 0.01:
                    drawn = overlay_fn(
                        frame, draw_mask, frame_idx=frame_idx, history=history,
                    )
                    if alpha >= 0.999:
                        frame = drawn
                    else:
                        frame = cv2.addWeighted(frame, 1 - alpha, drawn, alpha, 0)

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
