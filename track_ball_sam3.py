"""Highlight the ball in a soccer match video using SAM 3.1.

Phase 1 MVP: pre-recorded MP4 in, annotated MP4 out.

The SAM 3.1 video predictor returns per-frame outputs of the form:
    {
        "out_probs":         [p1, p2, ...],        # confidence per instance
        "out_boxes_xywh":    [[x,y,w,h], ...],     # normalized 0-1
        "out_obj_ids":       [id1, id2, ...],      # tracker IDs
        "out_binary_masks":  [mask1, mask2, ...],  # binary HxW per instance
    }

We union all detected-instance masks into a single binary mask and hand
it to one of the overlay functions in overlays.py.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import click
import cv2
import numpy as np
import torch
from tqdm import tqdm

from overlays import OVERLAYS


def build_predictor():
    """Construct the standard SAM 3 video predictor.

    We use SAM 3 (Nov 2025) not SAM 3.1 (Mar 2026): the public sam3.1_multiplex.pt
    checkpoint doesn't load on the public main branch — Meta trained it against
    an internal fork with renamed modules and didn't ship a converter. See
    facebookresearch/sam3 issue #526 (open as of 2026-05-26). The sibling
    facebook/sam3 / sam3.pt loads cleanly. For single-object tracking (one
    ball), there's no quality difference — multiplex is a multi-object
    throughput optimization, not a quality one.
    """
    from sam3.model_builder import build_sam3_video_predictor
    return build_sam3_video_predictor()


def _to_numpy_mask(m, target_hw: tuple[int, int]) -> np.ndarray:
    """Convert a single instance mask (torch tensor or ndarray) to uint8 HxW
    at target_hw, thresholded at 0.5."""
    if hasattr(m, "detach"):
        m = m.detach().cpu().numpy()
    m = np.asarray(m)
    if m.ndim > 2:
        m = m.squeeze()
    H, W = target_hw
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.uint8)


def _merge_masks(
    outputs: dict,
    frame_hw: tuple[int, int],
    sticky_id: int | None = None,
) -> tuple[np.ndarray | None, int | None]:
    """Pick a single per-instance mask and return (mask, picked_obj_id).

    Selection strategy (in order):
    1. If `sticky_id` was passed and is still in the current frame's IDs,
       keep tracking THAT object — SAM 3's tracker already maintains
       temporal identity, so locking onto the first frame's choice gives
       us stable single-ball tracking instead of jumping between IDs.
    2. Otherwise pick the highest-confidence detection (out_probs).
    3. If no probs, pick index 0.

    Returns (mask, picked_id) so the caller can thread sticky_id forward.
    Without stickiness, we saw the ring jump from the real ball to a
    player whenever a player briefly matched the "soccer ball" embedding
    with higher confidence than the actual ball.
    """
    masks = outputs.get("out_binary_masks", [])
    if len(masks) == 0:
        return None, None

    obj_ids_raw = outputs.get("out_obj_ids", [])
    obj_ids = [int(i) if hasattr(i, "item") else int(i) for i in obj_ids_raw]

    if sticky_id is not None and sticky_id in obj_ids:
        idx = obj_ids.index(sticky_id)
        return _to_numpy_mask(masks[idx], frame_hw), sticky_id

    probs = outputs.get("out_probs", [])
    if len(probs) == len(masks) and len(probs) > 0:
        probs_list = [float(p) if not hasattr(p, "item") else p.item() for p in probs]
        best_idx = int(np.argmax(probs_list))
    else:
        best_idx = 0

    picked_id = obj_ids[best_idx] if best_idx < len(obj_ids) else None
    return _to_numpy_mask(masks[best_idx], frame_hw), picked_id


def iter_predictions(
    predictor,
    video_path: Path,
    text_prompt: str,
) -> Iterator[tuple[np.ndarray, np.ndarray | None]]:
    """Yield (frame_bgr, mask_or_None) per frame, in chronological order.

    Assumes SAM 3.1's propagate_in_video stream emits one response per frame
    in order — true in the example notebook. If misaligned in practice, we'll
    notice by visible frame skew in the output.

    The forward pass runs under torch.autocast(bfloat16) because FlashAttention 3
    on Ada/Ampere GPUs only supports fp16/bf16 inputs. bf16 is preferred over
    fp16 for numerical stability (wider exponent range, lower overflow risk).
    """
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    with autocast:
        session = predictor.handle_request({
            "type": "start_session",
            "resource_path": str(video_path),
            # Keep decoded frames on CPU; only stream the current frame to GPU.
            # Critical for 8 GB consumer GPUs at 720p+.
            "offload_video_to_cpu": True,
        })
        session_id = session["session_id"]

        predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt,
        })

    cap = cv2.VideoCapture(str(video_path))
    sticky_id: int | None = None

    try:
        with autocast:
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
            })
            for response in stream:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                mask, sticky_id = _merge_masks(
                    response["outputs"], frame_bgr.shape[:2], sticky_id=sticky_id,
                )
                yield frame_bgr, mask
    finally:
        cap.release()
        try:
            predictor.handle_request({
                "type": "close_session",
                "session_id": session_id,
            })
        except Exception:
            pass


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
    help="SAM 3.1 text prompt for the object to track.",
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

    click.echo("Loading SAM 3 (standard video predictor)…")
    predictor = build_predictor()
    click.echo(f"Tracking prompt: {text!r}  overlay: {overlay}  frames: {n_frames}")

    t0 = time.perf_counter()
    n_done = 0
    n_hits = 0
    for frame, mask in tqdm(
        iter_predictions(predictor, input_path, text),
        total=n_frames,
        desc=overlay,
    ):
        if mask is not None and mask.any():
            frame = overlay_fn(frame, mask)
            n_hits += 1
        writer.write(frame)
        n_done += 1

    writer.release()
    elapsed = time.perf_counter() - t0
    fps_actual = n_done / elapsed if elapsed > 0 else 0.0
    click.echo(
        f"Wrote {output_path}\n"
        f"  frames={n_done} hits={n_hits} ({100*n_hits/max(n_done,1):.1f}%) "
        f"elapsed={elapsed:.1f}s avg_fps={fps_actual:.2f}"
    )


if __name__ == "__main__":
    main()
