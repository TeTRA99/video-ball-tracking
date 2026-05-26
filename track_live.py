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

import subprocess
import time
import wave
from collections import deque
from pathlib import Path

import click
import cv2
import numpy as np
from tqdm import tqdm

from overlays import OVERLAYS
from tracker import BallTracker
from track_ball import _select_best_mask, build_predictor, DEFAULT_BALL_CLASS_ID


def _resolve_audio_device(spec: str | None) -> int | None:
    """Map a CLI --audio-device value to a sounddevice input index.

    - None: system default input (whatever Windows currently selects)
    - integer string: use that exact device index
    - other string: case-insensitive substring match against device names
    """
    if spec is None:
        return None
    import sounddevice as sd

    if spec.isdigit():
        return int(spec)
    needle = spec.lower()
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and needle in d["name"].lower():
            return i
    lines = ["Audio input devices visible to sounddevice:"]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            lines.append(
                f"  [{i}] {d['name']} "
                f"({d['max_input_channels']} ch @ {int(d['default_samplerate'])} Hz)"
            )
    raise click.ClickException(
        f"No audio input device matched {spec!r}.\n" + "\n".join(lines)
    )


class _AudioCapture:
    """Streams audio from a USB UVC capture card's UAC sibling to a WAV.

    Float32 from sounddevice → int16 PCM in the WAV (the standard format
    for muxing back into MP4 via AAC encoding).
    """

    def __init__(
        self,
        device: int | None,
        samplerate: int,
        channels: int,
        wav_path: Path,
    ):
        import sounddevice as sd

        self._wav = wave.open(str(wav_path), "wb")
        self._wav.setnchannels(channels)
        self._wav.setsampwidth(2)
        self._wav.setframerate(samplerate)
        self._stream = sd.InputStream(
            device=device,
            samplerate=samplerate,
            channels=channels,
            callback=self._callback,
            blocksize=0,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            # Underruns / overflows print a warning but don't stop the stream.
            click.echo(f"audio stream status: {status}", err=True)
        pcm = (indata * 32767.0).clip(-32768, 32767).astype(np.int16)
        self._wav.writeframes(pcm.tobytes())

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()
        self._wav.close()


def _mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> tuple[bool, str]:
    """Mux a video-only MP4 + a WAV into a single MP4 via ffmpeg.

    -c:v copy avoids re-encoding the video (no quality loss, fast).
    -c:a aac re-encodes the WAV to AAC, which is the standard MP4 codec.
    -shortest trims to whichever stream is shorter — protects against the
    audio capture starting marginally before/after the video writer.
    Returns (success, ffmpeg stderr).
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


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
@click.option(
    "--audio-device",
    "audio_device_spec",
    type=str,
    default=None,
    help="Audio input for --record mux. None = system default. Integer = "
         "exact sounddevice index. String = case-insensitive name match "
         "(e.g. 'USB Audio', 'HDMI'). Pass --audio-device list to see "
         "available devices and exit.",
)
@click.option(
    "--no-audio",
    "skip_audio",
    is_flag=True,
    default=False,
    help="Skip audio capture even when --record is set (video-only MP4).",
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
    audio_device_spec: str | None,
    skip_audio: bool,
) -> None:
    """Stream a live source through the ball-tracking pipeline."""
    # Diagnostic: --audio-device list prints devices and exits without
    # spinning up any inference.
    if audio_device_spec == "list":
        import sounddevice as sd

        click.echo("Audio input devices:")
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                click.echo(
                    f"  [{i}] {d['name']} "
                    f"({d['max_input_channels']} ch @ {int(d['default_samplerate'])} Hz)"
                )
        return

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

    # Audio capture path: enabled when --record is set and --no-audio isn't.
    # Audio is captured to a temp WAV beside the requested record_path; after
    # the recording loop ends we mux video+audio into record_path via ffmpeg.
    capture_audio = record_path is not None and not skip_audio
    video_write_path: Path | None = None
    audio_write_path: Path | None = None
    audio_capture: _AudioCapture | None = None
    if record_path is not None:
        if capture_audio:
            video_write_path = record_path.with_suffix(record_path.suffix + ".video.mp4")
            audio_write_path = record_path.with_suffix(record_path.suffix + ".audio.wav")
        else:
            video_write_path = record_path

    def _open_writer(target_fps: float) -> cv2.VideoWriter:
        assert video_write_path is not None
        video_write_path.parent.mkdir(parents=True, exist_ok=True)
        # Codec selection: cv2.VideoWriter on Windows silently produces a
        # 0-byte file when the requested fourcc isn't available. Pick by
        # extension and try a small chain; MJPG/.avi is the always-works
        # fallback (bigger files, but reliable).
        ext = video_write_path.suffix.lower()
        if ext == ".avi":
            codec_chain = [("MJPG", "MJPG"), ("XVID", "XVID")]
        else:
            codec_chain = [("mp4v", "MPEG-4"), ("MJPG", "MJPG")]
        for fourcc_str, label in codec_chain:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            candidate = cv2.VideoWriter(str(video_write_path), fourcc, target_fps, (W, H))
            if candidate.isOpened():
                click.echo(
                    f"Recording video to: {video_write_path} "
                    f"({label}/{fourcc_str}) @ {target_fps:.1f}fps"
                )
                return candidate
            candidate.release()
        raise click.ClickException(
            f"No working codec found for {video_write_path}. "
            f"Try a .avi extension or install K-Lite Codec Pack."
        )

    def _start_audio_if_needed() -> None:
        """Open the audio capture stream once we're about to start writing
        video frames. Done lazily because the writer for screen sources
        opens only after the warmup completes — we want audio and video
        to start at the same wall-clock moment."""
        nonlocal audio_capture
        if not capture_audio or audio_capture is not None:
            return
        assert audio_write_path is not None
        try:
            device = _resolve_audio_device(audio_device_spec)
            audio_capture = _AudioCapture(
                device=device,
                samplerate=48000,
                channels=2,
                wav_path=audio_write_path,
            )
            audio_capture.start()
            click.echo(f"Recording audio to: {audio_write_path}")
        except Exception as exc:
            click.echo(f"Audio capture failed ({exc}); continuing video-only.", err=True)
            audio_capture = None

    writer: cv2.VideoWriter | None = None
    warmup_frames = 20 if (record_path and fs.kind == "screen") else 0
    if record_path and warmup_frames == 0:
        writer = _open_writer(src_fps)
        _start_audio_if_needed()

    tracker = BallTracker(
        max_jump_per_frame=max_jump_px,
        max_extrapolate_frames=max_extrapolate,
    )
    history: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    last_real_mask: np.ndarray | None = None
    last_real_centroid: tuple[int, int] | None = None
    sticky_id: int | None = None
    prev_draw_centroid: tuple[int, int] | None = None
    jump_fade_remaining = 0
    JUMP_FADE_FRAMES = 15
    JUMP_THRESHOLD_PX = 150

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
            mask, picked_id, picked_conf = _select_best_mask(
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
                    if prev_draw_centroid is not None:
                        dx = cx - prev_draw_centroid[0]
                        dy = cy - prev_draw_centroid[1]
                        if dx * dx + dy * dy > JUMP_THRESHOLD_PX * JUMP_THRESHOLD_PX:
                            jump_fade_remaining = JUMP_FADE_FRAMES
                    prev_draw_centroid = (cx, cy)

                    if jump_fade_remaining > 0:
                        alpha = 1.0 - (jump_fade_remaining / JUMP_FADE_FRAMES)
                        jump_fade_remaining -= 1
                    else:
                        alpha = 1.0

                    if alpha > 0.01:
                        drawn = overlay_fn(
                            frame, draw_mask, frame_idx=n_frames, history=history,
                        )
                        if alpha >= 0.999:
                            frame = drawn
                        else:
                            frame = cv2.addWeighted(frame, 1 - alpha, drawn, alpha, 0)

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
                _start_audio_if_needed()

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
        if audio_capture is not None:
            try:
                audio_capture.stop()
            except Exception as exc:
                click.echo(f"audio_capture.stop() raised: {exc}", err=True)
        if window:
            cv2.destroyAllWindows()

    # Mux audio + video into the requested record_path. Done after the
    # try/finally so any stop errors above don't leave us holding open
    # files when ffmpeg tries to read them.
    if (
        capture_audio
        and audio_capture is not None
        and record_path is not None
        and video_write_path is not None
        and audio_write_path is not None
        and video_write_path.exists()
        and audio_write_path.exists()
        and audio_write_path.stat().st_size > 0
    ):
        click.echo(f"Muxing into {record_path}…")
        ok, stderr = _mux_video_audio(video_write_path, audio_write_path, record_path)
        if ok:
            video_write_path.unlink(missing_ok=True)
            audio_write_path.unlink(missing_ok=True)
            click.echo(f"Wrote {record_path} (video + audio)")
        else:
            click.echo(
                "ffmpeg mux failed — keeping the temp files so you can mux manually:\n"
                f"  video: {video_write_path}\n  audio: {audio_write_path}\n"
                f"ffmpeg stderr (last lines):\n{stderr[-2000:]}",
                err=True,
            )

    elapsed = time.perf_counter() - t_start
    avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
    click.echo(
        f"Stopped. frames={n_frames} elapsed={elapsed:.1f}s "
        f"avg_fps={avg_fps:.2f}"
    )


if __name__ == "__main__":
    main()
