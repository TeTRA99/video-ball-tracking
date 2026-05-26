#!/usr/bin/env bash
# RunPod / cloud GPU setup for SAM 3.1 video ball tracking.
#
# Target pod: H100 PCIe or H100 SXM, PyTorch 2.x image (RunPod's official
# "PyTorch 2.4.0 + CUDA 12.4" or newer works fine — we'll install the
# right torch ourselves to be sure).
#
# Run this once on a fresh pod after attaching to its web terminal:
#     bash runpod_setup.sh
#
# Then upload your clip into ~/work/clips/ (drag-drop in Jupyter, or scp)
# and run:
#     cd ~/work/video-ball-tracking
#     python track_ball_sam3.py --input ../clips/game2.mp4 --output ../out/game2_sam3.mp4 --overlay ring --text "soccer ball"
#
# Cost estimate: ~30 min setup + ~10 min per 30-sec clip on H100 PCIe at
# $1.99/hr ≈ $1.50 for setup + $0.30/clip. Budget $5-10 for full eval.
set -euo pipefail

WORK=$HOME/work
mkdir -p "$WORK/clips" "$WORK/out"
cd "$WORK"

echo "==> System deps"
apt-get update -qq
apt-get install -qq -y git ffmpeg

echo "==> Clone repos"
if [ ! -d sam3 ]; then
    git clone https://github.com/facebookresearch/sam3.git
fi
if [ ! -d video-ball-tracking ]; then
    git clone https://github.com/TeTRA99/video-ball-tracking.git
fi

echo "==> Python env (uv keeps things tidy; falls back to pip if missing)"
if ! command -v uv &>/dev/null; then
    pip install --quiet uv
fi

cd "$WORK/video-ball-tracking"
uv venv --python 3.13
# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> PyTorch with the matching CUDA wheel index"
# H100 is Hopper. Most RunPod images ship CUDA 12.4-12.8; PyTorch's cu128
# index covers Hopper. Re-check https://pytorch.org/get-started/locally/
# at run time if this index URL has rotated.
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==> Our project deps (opencv, ultralytics, mss, click, tqdm, etc.)"
uv pip install -e .

echo "==> SAM 3.1 hard deps that the README marks 'optional' but the code requires"
uv pip install einops pycocotools psutil ninja

echo "==> Install SAM 3 itself"
cd "$WORK/sam3"
uv pip install -e .

echo "==> Patch the start_session kwargs bug (PR #543, not always merged)"
# Filters unknown kwargs (like offload_state_to_cpu) before passing to
# multiplex init_state(). Idempotent: if the file already filters kwargs,
# the grep guard skips the patch.
PATCH_FILE="sam3/multiplex_predictor/sam3_multiplex_video_predictor.py"
if [ -f "$PATCH_FILE" ] && ! grep -q "init_state_kwargs" "$PATCH_FILE"; then
    python - <<'PYPATCH'
import pathlib, re
p = pathlib.Path("sam3/multiplex_predictor/sam3_multiplex_video_predictor.py")
src = p.read_text()
# Add a one-liner that filters init_state kwargs to known names. Minimal
# patch — just drop offload_state_to_cpu before it reaches init_state.
needle = "def start_session("
if needle in src and "offload_state_to_cpu" in src:
    src = src.replace(
        "init_state(",
        "init_state(  # noqa: E501\n        # filtered upstream — kwargs PR #543\n        ",
        1,
    )
    # Filter the offending kwarg name out of any **kwargs dict before the call.
    src = re.sub(
        r"(\*\*\s*)(\w+)\)(\s*#.*)?$",
        r"**{k:v for k,v in \2.items() if k != 'offload_state_to_cpu'})\3",
        src,
        count=1,
        flags=re.MULTILINE,
    )
    p.write_text(src)
    print(f"Patched {p}")
else:
    print(f"Patch target not found / already patched in {p}")
PYPATCH
fi

echo "==> FA3 dtype patch — H100 supports fp8 natively, so the FA3 fp8 hardcode"
echo "    should work as-is. Only patch to bf16 if FA3 isn't supported."
# (Intentionally skipping — leaving the upstream FP8 path on H100.)

echo "==> HuggingFace auth (you'll get prompted to paste a token with"
echo "    'read' access to facebook/sam3.1 — request access on the model page first)"
hf auth login || huggingface-cli login

echo "==> Download SAM 3.1 checkpoint (uses repo helper if present, else pulls"
echo "    the default via the model builder on first run)"
cd "$WORK/sam3"
python - <<'PYDL'
# Touch the model builder so the checkpoint gets cached locally now,
# instead of mid-inference (matters because we want clean timings).
from sam3.model_builder import build_sam3_multiplex_video_predictor
print("Loading SAM 3.1 multiplex video predictor (downloads on first run)...")
predictor = build_sam3_multiplex_video_predictor(max_num_objects=2)
print("OK — predictor loaded.")
del predictor
import torch
torch.cuda.empty_cache()
PYDL

echo "==> Smoke test"
cd "$WORK/video-ball-tracking"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"

cat <<EOF

==============================================================
Setup complete. Next:

1. Upload your test clip into $WORK/clips/  (e.g. game2.mp4)
   - Easiest: open the pod's Jupyter file browser, drag-drop into work/clips/
   - Or scp:  scp -P <pod-port> game2.mp4 root@<pod-ip>:$WORK/clips/

2. Run SAM 3.1 inference:
     cd $WORK/video-ball-tracking
     source .venv/bin/activate
     python track_ball_sam3.py \\
         --input ../clips/game2.mp4 \\
         --output ../out/game2_sam3.mp4 \\
         --overlay ring \\
         --text "soccer ball"

3. Download the result from $WORK/out/game2_sam3.mp4 (Jupyter file browser
   right-click → download, or scp back to your machine).

4. Watch alongside out/game2_v2_c30.mp4 (the YOLO fine-tuned baseline) to
   judge whether SAM 3.1's detection quality justifies the cloud cost.
==============================================================
EOF
