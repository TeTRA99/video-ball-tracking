# Video ball tracking

Highlights the ball in a soccer match video so people with low vision can follow the play. Uses Meta's **SAM 3.1** for text-prompted video segmentation, then renders a configurable overlay (ring / halo / arrow / recolor) on top of the ball's mask.

Full plan: `~/.claude/plans/i-want-you-to-lively-avalanche.md`.

## Status

**Phase 1 — MP4 in → annotated MP4 out.** Scaffolding committed. The SAM 3 video iteration loop in [track_ball.py](track_ball.py) is a `TODO` — wire it up on the GPU machine once SAM 3.1 is installed.

## Hardware target

RTX 4070 PC (Windows + WSL2 + Ubuntu). Mac is for editing only — SAM 3 is CUDA-only and won't run natively on Apple Silicon.

## Setup (do this on the GPU machine, inside WSL2)

**Re-verify every tool's current stable version on its own homepage before installing.** This snapshot is from May 2026 and the AI/ML stack moves fast.

1. **Host driver + CUDA.** Install the latest NVIDIA driver supporting the current CUDA Toolkit (13.2.1 was current as of 2026-04). `nvidia-smi` should work in both Windows and inside WSL2.
2. **Toolchain inside WSL2.** Install [`uv`](https://docs.astral.sh/uv/), `git`, and `ffmpeg`.
3. **Clone repos.** This project and SAM 3 upstream side by side:
   ```bash
   git clone <this-repo> video-ball-tracking
   git clone https://github.com/facebookresearch/sam3.git
   ```
4. **Create env and install.** Check the current PyTorch CUDA wheel index URL at <https://pytorch.org/get-started/locally/> before running:
   ```bash
   cd video-ball-tracking
   uv venv --python 3.13
   source .venv/bin/activate
   uv pip install --pre torch --index-url https://download.pytorch.org/whl/cu128
   uv pip install -e ../sam3
   uv pip install -e .
   ```
5. **Hugging Face access.** Request access to `facebook/sam3.1` on the model page, then:
   ```bash
   huggingface-cli login
   ```
6. **Smoke test.**
   ```bash
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

## Usage

```bash
python track_ball.py \
  --input clips/match.mp4 \
  --output out/match_ring.mp4 \
  --overlay ring \
  --text "soccer ball"
```

Available overlays: `ring`, `halo`, `arrow`, `recolor`. See [overlays.py](overlays.py) — each is a small pure function `(frame, mask) -> frame`, easy to tweak or add new styles.

## Project layout

- [track_ball.py](track_ball.py) — CLI entrypoint and per-frame pipeline
- [overlays.py](overlays.py) — pluggable overlay functions
- [pyproject.toml](pyproject.toml) — Python dependencies (excludes SAM 3 itself; that comes from the cloned upstream)
- `clips/` — input videos (gitignored)
- `out/` — annotated outputs (gitignored)
- `weights/` — model checkpoints (gitignored)
