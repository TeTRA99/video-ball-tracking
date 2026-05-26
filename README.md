# Video ball tracking

Highlights the ball in a soccer match video so people with low vision can follow the play. Real-time live overlay on a consumer GPU.

**v1 status (2026-05-26):** working end-to-end on RTX 4070 Laptop at ~45 FPS. Live broadcast input via a USB 3.0 HDMI capture card. Offline file → annotated file works too.

## Pipeline

```
HDMI source (set-top box, streaming stick)
  ↓
USB 3.0 HDMI capture card (UVC / plug-and-play, shows up as a webcam)
  ↓
track_live.py  →  YOLO26s (fine-tuned on soccer)
              →  ByteTrack id association (tuned bytetrack_ball.yaml)
              →  Kinematic smoothing + 3-frame extrapolation
              →  Ring overlay with 15-frame fade after teleport-sized jumps
              →  cv2.imshow window  (and/or  --record out.mp4)
```

## Hardware

- **Compute:** Windows 11 + RTX 4070 Laptop (8 GB VRAM dedicated, 8 GB shared). Native Windows Python preferred over WSL2 — WSL's OpenCV doesn't ship a GUI backend and screen-capture can't see Windows desktop pixels.
- **Live input:** any USB 3.0 + UVC HDMI capture card (search "HDMI USB 3.0 1080p60 UVC plug and play", ~$25-40). Avoid USB 2.0 devices — meaningfully higher latency.

The Mac is for code editing only.

## Setup (Windows native, not WSL)

```powershell
# 1. Install Python 3.12 (NOT 3.13 — torch 2.11+cu128 has parse bugs under 3.13)
# Download from python.org, check "Add to PATH" during install.

# 2. Clone
cd $HOME\dev
git clone https://github.com/TeTRA99/video-ball-tracking.git
cd video-ball-tracking

# 3. Venv + project deps (uses the venv's python directly to sidestep
#    PowerShell execution-policy and Microsoft Store alias issues)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .

# 4. PyTorch with CUDA (cu128 matches Hopper/Ada; for older host drivers
#    try cu126 or cu124)
.venv\Scripts\python.exe -m pip install --pre --upgrade --force-reinstall `
  torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 5. Verify
.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On the Mac:

```bash
cd "Documents/Claude Work/Video ball tracking test"
uv venv --python 3.13   # Mac venv can use 3.13; torch on MPS is fine
uv pip install -e .
```

## Usage

**Live mode (the main MVP):**

```powershell
# Once your HDMI capture card is connected and showing the broadcast:
.venv\Scripts\python.exe track_live.py --source 1 `
  --model runs\detect\runs\ball_finetune_v2_1280\weights\best.pt `
  --ball-class 0 --overlay ring --conf 0.30 --imgsz 640 `
  --tracker trackers/bytetrack_ball.yaml
```

`--source` is the camera index — try 0, 1, 2 until you see the broadcast feed. Press `q` in the preview window to quit. Add `--record out\live.mp4` to also save.

**Other live sources:**
- `--source 0` — built-in webcam (Mac/laptop)
- `--source rtsp://...` — IP camera / network stream
- `--screen` — capture the primary monitor (Windows native only; doesn't work in WSL because WSL can't see the Windows compositor)
- `--source clips/test.mp4` — a video file (useful for testing the live loop without hardware)

**Offline file → annotated file:**

```powershell
.venv\Scripts\python.exe track_ball.py --input clips/game.mp4 --output out/game.mp4 `
  --overlay ring `
  --model runs\detect\runs\ball_finetune_v2_1280\weights\best.pt `
  --ball-class 0 --conf 0.30 --imgsz 1280 `
  --tracker trackers/bytetrack_ball.yaml
```

**Overlay styles** — `ring` (default, recommended), `halo`, `arrow`, `recolor`, `pulse`, `chevron`. See [overlays.py](overlays.py).

## Fine-tuning your own weights

If the bundled fine-tuned weights aren't included (they're gitignored):

```bash
# On the Windows machine inside WSL or native, with CUDA available:
python datasets/download.py        # pulls Roboflow football-players-detection
python train.py                    # ~30-45 min on RTX 4070 Laptop at imgsz=1280
# Resulting best.pt is at runs/detect/runs/ball_finetune_v2_1280/weights/best.pt
```

## What we learned, what we tried, what we ruled out

- **SAM 3 / 3.1 for live: ruled out.** Architecturally batch-only (needs entire video upfront, bidirectional context), ~3 FPS on L40S even after fixing the mask-selection bugs. Cloud-only economics ($600-2400/month per user) don't fit accessibility distribution. **SAM 3.1 multiplex checkpoint also broken on public code** (facebookresearch/sam3 [issue #526](https://github.com/facebookresearch/sam3/issues/526)). Track_ball_sam3.py preserved for future use if a streaming SAM variant ships.
- **Larger YOLO variants (-l, -x): tested, diminishing returns** vs fine-tuned -s.
- **More fine-tuning data: not pursued.** 372 images was thin but improvements past v2 (imgsz=1280) plateaued without obvious wins on real clips.
- **RF-DETR Soccernet: not tested.** Listed as plausible alternative to YOLO; ~3-4 hours of work to wire in. Has custom API (returns pandas DataFrames), trained specifically on broadcast soccer.

## When to revisit SAM 3.x for live

Any of these would unlock revisiting:

1. **Streaming / causal video predictor** — SAM 3.x currently uses bidirectional context; a frame-by-frame causal variant would unlock live. No public roadmap.
2. **[EfficientSAM3 Stage 2 weights](https://github.com/SimonZeng7108/efficientsam3)** — distilled video tracker. Stage 1 (image encoder) shipped; Stage 2 in development. When Stage 2 ships, expect ~10× smaller models for consumer-GPU real-time.
3. **Ultralytics integration with real-time wrappers** — if `docs.ultralytics.com/models/sam-3` adds a `track(stream=True)` mode the way they did for YOLO, someone has solved the streaming wrap.

Concrete trigger to act: a SAM 3.x variant running streaming, at 15+ FPS sustained, on <16 GB VRAM consumer hardware.

## Project layout

- [track_ball.py](track_ball.py) — offline file → annotated file
- [track_live.py](track_live.py) — live webcam / RTSP / screen → annotated stream
- [track_ball_sam3.py](track_ball_sam3.py) — SAM 3 video predictor (for cloud H100/L40S use; preserved)
- [overlays.py](overlays.py) — pluggable overlay functions
- [tracker.py](tracker.py) — kinematic ball tracker (smoothing + extrapolation)
- [train.py](train.py) — fine-tuning loop (Ultralytics YOLO26 on Roboflow football-players-detection)
- [trackers/bytetrack_ball.yaml](trackers/bytetrack_ball.yaml) — tuned ByteTrack config for soccer ball tracking
- [runpod_setup.sh](runpod_setup.sh) — cloud GPU bootstrap for SAM 3 evaluation
- [pyproject.toml](pyproject.toml) — Python deps
- `clips/`, `out/`, `weights/`, `runs/` — gitignored data dirs
