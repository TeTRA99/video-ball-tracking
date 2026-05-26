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


class _FrameSource:
    """Unified BGR frame producer for the live loop.

    Wraps either cv2.VideoCapture (file path, webcam index, RTSP/HTTP URL)
    or mss screen capture behind a single .read() / .release() interface.
    Hides the awkwardness of bootstrapping the first frame to learn HxW
    before the main loop starts.
    """

    def __init__(
        self,
        cv_source: int | str | None = None,
        screen_region: tuple[int, int, int, int] | None = None,
    ):
        self.kind = "screen" if cv_source is None else "cv"
        self.total: int | None = None

        if self.kind == "screen":
            import mss

            self._sct = mss.mss()
            # monitors[0] is the union of all screens; [1] is the primary.
            primary = self._sct.monitors[1]
            if screen_region is not None:
                x, y, w, h = screen_region
                self._monitor = {"left": x, "top": y, "width": w, "height": h}
            else:
                self._monitor = primary
            self.fps = 30.0  # nominal; mss is fast enough not to bottleneck
            # Do a trial grab so H/W reflect the ACTUAL captured pixel
            # dimensions, not the monitor metadata. On Windows with DPI
            # scaling != 100%, mss.grab returns physical pixels even
            # though monitors[].width reports logical. Mismatch makes
            # cv2.VideoWriter silently drop every frame.
            shot = self._sct.grab(self._monitor)
            self._pending = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
            self.H, self.W = self._pending.shape[:2]
            return

        self._cap = cv2.VideoCapture(cv_source)
        if not self._cap.isOpened():
            raise click.ClickException(f"Could not open source: {cv_source}")
        # Cuts capture-side lag when the producer is faster than inference.
        # Silently ignored by backends that don't honor it.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = self._cap.read()
        if not ret:
            self._cap.release()
            raise click.ClickException("Failed to read first frame from source.")
        self.H, self.W = frame.shape[:2]
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        self._pending = frame

    def read(self) -> np.ndarray | None:
        # Both modes prime _pending with the first frame in __init__
        # (cv to learn HxW + FPS, screen to learn actual pixel dims under
        # DPI scaling). Hand that back on the first call, then take the
        # appropriate fresh-frame path.
        if self._pending is not None:
            f, self._pending = self._pending, None
            return f
        if self.kind == "screen":
            shot = self._sct.grab(self._monitor)
            return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self) -> None:
        if self.kind == "screen":
            self._sct.close()
        else:
            self._cap.release()


@click.command()
@click.option(
    "--source",
    default="0",
    show_default=True,
    help="Webcam index (0, 1, ...), file path, or RTSP/HTTP URL. "
         "Ignored when --screen is set.",
)
@click.option(
    "--screen",
    is_flag=True,
    default=False,
    help="Capture from the screen instead of --source. Useful for "
         "demoing the live pipeline against any video playing on the "
         "desktop (browser, VLC, capture-card app).",
)
@click.option(
    "--screen-region",
    default=None,
    help="Limit screen capture to a region: 'X,Y,W,H' in screen pixels. "
         "Default is the full primary monitor. Useful for tracking just "
         "a video player window instead of the whole screen.",
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
@click.option(
    "--tracker",
    "tracker_yaml",
    type=str,
    default="bytetrack.yaml",
    show_default=True,
    help="Built-in: 'bytetrack.yaml' or 'botsort.yaml'. Or a custom path "
         "like 'trackers/bytetrack_ball.yaml' (shipped, tuned for ball).",
)
def main(
    source: str,
    screen: bool,
    screen_region: str | None,
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
    tracker_yaml: str,
) -> None:
    """Stream a live source through the ball-tracking pipeline."""
    overlay_fn = OVERLAYS[overlay]

    if model_path:
        click.echo(f"Loading custom weights: {model_path}")
    else:
        click.echo(f"Loading yolo26{model_size}-seg…")
    predictor = build_predictor(
        model_size,
        weights_path=str(model_path) if model_path else None,
    )

    if screen:
        region: tuple[int, int, int, int] | None = None
        if screen_region:
            try:
                parts = [int(p) for p in screen_region.split(",")]
                if len(parts) != 4:
                    raise ValueError
                region = (parts[0], parts[1], parts[2], parts[3])
            except ValueError as exc:
                raise click.ClickException(
                    f"--screen-region must be 'X,Y,W,H', got {screen_region!r}"
                ) from exc
        fs = _FrameSource(cv_source=None, screen_region=region)
        src_label = f"screen[{fs.W}x{fs.H}]"
    else:
        cv_source: int | str = int(source) if source.isdigit() else source
        fs = _FrameSource(cv_source=cv_source)
        src_label = str(cv_source)
    H, W, src_fps, total_frames = fs.H, fs.W, fs.fps, fs.total

    def _open_writer(target_fps: float) -> cv2.VideoWriter:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        # Codec selection: cv2.VideoWriter on Windows silently produces a
        # 0-byte file when the requested fourcc isn't available. Pick by
        # extension and try a small chain; MJPG/.avi is the always-works
        # fallback (bigger files, but reliable).
        ext = record_path.suffix.lower()
        if ext == ".avi":
            codec_chain = [("MJPG", "MJPG"), ("XVID", "XVID")]
        else:
            # Skip avc1/H.264 even though it'd produce smaller files —
            # Windows cv2.VideoWriter returns isOpened()=True even when
            # OpenH264 is missing and ffmpeg's encoder init silently
            # fails, so we'd "succeed" into a non-writing writer.
            # mp4v is Windows-native and works without extra libraries.
            codec_chain = [("mp4v", "MPEG-4"), ("MJPG", "MJPG")]
        for fourcc_str, label in codec_chain:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            candidate = cv2.VideoWriter(str(record_path), fourcc, target_fps, (W, H))
            if candidate.isOpened():
                click.echo(
                    f"Recording to: {record_path} ({label}/{fourcc_str}) "
                    f"@ {target_fps:.1f}fps"
                )
                return candidate
            candidate.release()
        raise click.ClickException(
            f"No working codec found for {record_path}. "
            f"Try a .avi extension or install K-Lite Codec Pack."
        )

    # For file/webcam/URL sources, src_fps is trustworthy (cv2 reports it).
    # For screen capture it's a nominal 30 — actual capture rate depends
    # on the inference loop and varies. We open the writer for non-screen
    # sources immediately, but defer screen-source writer opening until
    # we've measured the actual loop rate via a warmup, so playback
    # speed matches reality.
    writer: cv2.VideoWriter | None = None
    warmup_frames = 20 if (record_path and fs.kind == "screen") else 0
    if record_path and warmup_frames == 0:
        writer = _open_writer(src_fps)

    tracker = BallTracker(
        max_jump_per_frame=max_jump_px,
        max_extrapolate_frames=max_extrapolate,
    )
    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    last_real_mask: np.ndarray | None = None
    last_real_centroid: tuple[int, int] | None = None
    sticky_id: int | None = None

    click.echo(
        f"Live tracking — source={src_label} {W}x{H}@{src_fps:.1f}fps "
        f"overlay={overlay} conf>={conf} imgsz={imgsz} tracker={tracker_yaml}  "
        f"press 'q' or ESC in the window to quit"
    )

    t_start = time.perf_counter()
    n_frames = 0
    rolling_fps = 0.0
    t_last = t_start
    pbar = tqdm(total=total_frames, desc="live", unit="f")

    try:
        while True:
            frame = fs.read()
            if frame is None:
                click.echo("Source ended.")
                break

            # persist=True keeps ByteTrack state across single-frame calls.
            results = predictor.track(
                frame,
                classes=[ball_class],
                conf=conf,
                imgsz=imgsz,
                tracker=tracker_yaml,
                persist=True,
                verbose=False,
            )
            result = results[0]
            mask, picked_id = _select_best_mask(
                result, frame.shape[:2],
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
            elif record_path and n_frames + 1 == warmup_frames:
                # Warmup complete — open the writer with measured rate.
                # rolling_fps has converged on actual loop throughput.
                measured = rolling_fps if rolling_fps > 0 else src_fps
                writer = _open_writer(measured)

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
        fs.release()
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
