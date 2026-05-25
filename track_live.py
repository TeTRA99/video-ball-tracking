"""Live ball tracking from a webcam (or any cv2.VideoCapture source).

Phase 4 entrypoint — same detection / tracker / overlay stack as the
file-based pipeline, but reads frames from a live source and displays
them in an OpenCV window in real time. Press 'q' or ESC to quit.

Typical use (webcam pointed at a TV showing a match), fine-tuned weights:
    python track_live.py \\
        --model runs/detect/runs/ball_finetune_v2_1280/weights/best.pt \\
        --ball-class 0 \\
        --overlay ring

If your webcam is not index 0, try `--source 1` (or higher). The flag
also accepts an RTSP/HTTP URL — useful later for IP cameras or screen-
capture pipelines. Pass `--record out/live.mp4` to also save the stream.

Latency notes:
- imgsz=640 keeps inference under ~50 ms on the 4070 Laptop, giving us
  ~15-25 effective FPS end-to-end. Drop to 480 if the source is 720p+
  and you need more headroom.
- cv2.VideoCapture buffers a few frames internally. For low-latency
  live overlay, we drain to the latest frame each loop iteration.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from overlays import OVERLAYS
from tracker import BallTracker
from track_ball import _select_best_mask, build_predictor, DEFAULT_BALL_CLASS_ID


@click.command()
@click.option(
    "--source",
    default="0",
    show_default=True,
    help="Webcam index (0, 1, ...) or RTSP/HTTP URL.",
)
@click.option(
    "--overlay",
    type=click.Choice(list(OVERLAYS)),
    default="ring",
    show_default=True,
)
@click.option(
    "--model-size",
    type=click.Choice(["n", "s", "m", "l", "x"]),
    default="n",
    show_default=True,
    help="Stock YOLO26-seg size when --model is not given.",
)
@click.option(
    "--model",
    "model_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Custom weights (e.g. runs/.../best.pt). Overrides --model-size.",
)
@click.option(
    "--ball-class",
    type=int,
    default=DEFAULT_BALL_CLASS_ID,
    show_default=True,
    help="32 for stock YOLO26 (COCO sports ball), 0 for fine-tuned.",
)
@click.option("--conf", type=float, default=0.25, show_default=True)
@click.option(
    "--imgsz",
    type=int,
    default=640,
    show_default=True,
    help="Lower = faster live throughput. 640 is a good 4070 default.",
)
@click.option("--max-ball-px", type=int, default=40, show_default=True)
@click.option("--max-jump-px", type=int, default=150, show_default=True)
@click.option(
    "--max-extrapolate",
    type=int,
    default=3,
    show_default=True,
    help="Bridge brief detection misses with constant-velocity prediction. "
         "1-3 frames is invisible smoothing; past ~3 the drift becomes "
         "noticeable in real soccer. Set 0 to disable.",
)
@click.option("--trail-frames", type=int, default=30, show_default=True)
@click.option(
    "--record",
    "record_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional: also save the annotated stream as MP4.",
)
@click.option(
    "--window/--no-window",
    default=True,
    show_default=True,
    help="Show a preview window. Disable for headless recording.",
)
def main(
    source: str,
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
    record_path: Path | None,
    window: bool,
) -> None:
    """Stream a live source through the ball-tracking pipeline."""
    src: int | str = int(source) if source.isdigit() else source
    overlay_fn = OVERLAYS[overlay]

    if model_path:
        click.echo(f"Loading custom weights: {model_path}")
    else:
        click.echo(f"Loading yolo26{model_size}-seg…")
    predictor = build_predictor(
        model_size,
        weights_path=str(model_path) if model_path else None,
    )

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise click.ClickException(f"Could not open source: {source}")

    # Reduce capture buffer to minimize lag when the producer (webcam /
    # capture card) is faster than our inference loop. Not all backends
    # honor this — silently ignored if unsupported.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise click.ClickException("Failed to read first frame from source.")
    H, W = frame.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 0 / -1 for live sources (webcam, RTSP). tqdm shows just the count
    # in that case, which is fine.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    writer = None
    if record_path:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(record_path), fourcc, src_fps, (W, H))
        click.echo(f"Recording to: {record_path}")

    tracker = BallTracker(
        max_jump_per_frame=max_jump_px,
        max_extrapolate_frames=max_extrapolate,
    )
    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    last_real_mask: np.ndarray | None = None
    last_real_centroid: tuple[int, int] | None = None

    click.echo(
        f"Live tracking — source={src} {W}x{H}@{src_fps:.1f}fps "
        f"overlay={overlay} conf>={conf} imgsz={imgsz}  "
        f"press 'q' or ESC in the window to quit"
    )

    t_start = time.perf_counter()
    n_frames = 0
    rolling_fps = 0.0
    t_last = t_start
    first_iter = True
    pbar = tqdm(total=total_frames, desc="live", unit="f")

    try:
        while True:
            if not first_iter:
                ret, frame = cap.read()
                if not ret:
                    click.echo("Source ended.")
                    break
            first_iter = False

            results = predictor(
                frame,
                classes=[ball_class],
                conf=conf,
                imgsz=imgsz,
                verbose=False,
            )
            result = results[0]
            mask = _select_best_mask(
                result, frame.shape[:2],
                max_ball_px=max_ball_px,
                prev_centroid=tracker.last_centroid,
            )

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
                else:
                    if last_real_mask is not None and last_real_centroid is not None:
                        dx = cx - last_real_centroid[0]
                        dy = cy - last_real_centroid[1]
                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                        draw_mask = cv2.warpAffine(
                            last_real_mask, M,
                            (last_real_mask.shape[1], last_real_mask.shape[0]),
                            flags=cv2.INTER_NEAREST, borderValue=0,
                        )
                    else:
                        draw_mask = None

                if draw_mask is not None:
                    frame = overlay_fn(
                        frame, draw_mask, frame_idx=n_frames, history=history,
                    )

            # Rolling FPS in the HUD so you can see live perf
            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            rolling_fps = (
                0.9 * rolling_fps + 0.1 * inst_fps if n_frames > 0 else inst_fps
            )
            cv2.putText(
                frame, f"{rolling_fps:.1f} fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if window:
                cv2.imshow("ball tracker (q/ESC to quit)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            n_frames += 1
            pbar.update(1)
            if n_frames % 10 == 0:
                pbar.set_postfix(fps=f"{rolling_fps:.1f}")
    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if window:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
    click.echo(
        f"Stopped. frames={n_frames} elapsed={elapsed:.1f}s "
        f"avg_fps={avg_fps:.2f}"
    )


if __name__ == "__main__":
    main()
