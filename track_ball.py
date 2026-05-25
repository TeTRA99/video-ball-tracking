"""Highlight the ball in a soccer match video using SAM 3.1.

Phase 1 MVP: pre-recorded MP4 in, annotated MP4 out.

This file is a runnable skeleton. The SAM 3 video iteration loop is marked
with TODOs — finalize against the upstream API once the model is installed
on the GPU machine. Verify the API at https://github.com/facebookresearch/sam3
before relying on the snippets below; it has changed between releases.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import click
import cv2
import numpy as np
from tqdm import tqdm

from overlays import OVERLAYS


def build_predictor():
    """Construct a SAM 3.1 video predictor.

    Lazy import so this module can be scaffolded on machines without a CUDA
    PyTorch build (e.g. development on macOS).
    """
    from sam3.video import build_sam3_video_predictor  # noqa: not on PyPI
    return build_sam3_video_predictor()


def iter_predictions(
    predictor,
    video_path: Path,
    text_prompt: str,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Yield (frame_bgr, mask_or_None) for each frame of the video.

    TODO Phase 1: implement the SAM 3 video session loop. The README at
    https://github.com/facebookresearch/sam3 shows the request/response
    handshake — typically:
        1. start_session(resource_path=video_path)
        2. add_prompt(frame_index=0, text=text_prompt)
        3. propagate / step through frames, receiving masks per frame
    Wire the per-frame frames + masks here. Use mask=None when the model
    returns no instance for that frame.
    """
    raise NotImplementedError(
        "Phase 1 TODO — wire SAM 3 video predictor against upstream API."
    )


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
    "--text",
    default="soccer ball",
    show_default=True,
    help="SAM 3 text prompt for the object to track.",
)
def main(input_path: Path, output_path: Path, overlay: str, text: str) -> None:
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

    predictor = build_predictor()

    t0 = time.perf_counter()
    n_done = 0
    for frame, mask in tqdm(
        iter_predictions(predictor, input_path, text),
        total=n_frames,
        desc=f"{overlay}",
    ):
        if mask is not None:
            frame = overlay_fn(frame, mask)
        writer.write(frame)
        n_done += 1

    writer.release()

    elapsed = time.perf_counter() - t0
    fps_actual = n_done / elapsed if elapsed > 0 else 0.0
    click.echo(
        f"Wrote {output_path} — {n_done} frames in {elapsed:.1f}s "
        f"({fps_actual:.2f} FPS)"
    )


if __name__ == "__main__":
    main()
