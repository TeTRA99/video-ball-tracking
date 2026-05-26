#!/usr/bin/env bash
# RunPod / cloud GPU setup for SAM 3 video ball tracking.
#
# NOTE: this runs SAM 3 (Nov 2025), not SAM 3.1 (Mar 2026). The public
# sam3.1_multiplex.pt checkpoint is broken on public-main code (Meta
# trained it against an internal fork; no migration shipped). See
# facebookresearch/sam3 issue #526, still open as of 2026-05-26. For
# single-object tracking like ours, SAM 3 and SAM 3.1 are quality-
# equivalent — multiplex is a multi-object throughput optimization.
#
# Recommended pod: A100 80GB Community Cloud (~$1/hr). Has plenty of
# headroom for SAM 3 (~20 GB peak), half the H100 price. H100 PCIe
# (~$2/hr) also works if A100 isn't available in your region.
#
# Run this once on a fresh pod after attaching to its web terminal:
#     bash runpod_setup.sh
#
# Then upload your clip into ~/work/clips/ (drag-drop in Jupyter, or scp)
# and run:
#     cd ~/work/video-ball-tracking
#     source .venv/bin/activate
#     python track_ball_sam3.py --input ../clips/game2.mp4 \
#         --output ../out/game2_sam3.mp4 --overlay ring --text "soccer ball"
#
# Cost estimate: ~30 min setup + ~10-15 min per 30-sec clip on A100 80GB
# at $1/hr ≈ $0.50 setup + $0.20/clip. Budget $2-4 for full eval.
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

echo "==> Detect host CUDA version and pick the matching PyTorch wheel index"
# RunPod templates ship anywhere from CUDA 12.1 to 13.x. Installing a torch
# wheel built against a NEWER CUDA than the host driver supports yields the
# "CUDA driver initialization failed" trap. Detect and match.
HOST_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 | awk '{print $3}')
echo "    Host CUDA: ${HOST_CUDA:-unknown}"
case "$HOST_CUDA" in
    12.1*|12.2*|12.3*|12.4*) WHEEL_IDX="cu124" ;;
    12.5*|12.6*)             WHEEL_IDX="cu126" ;;
    12.7*|12.8*|12.9*|13.*)  WHEEL_IDX="cu128" ;;
    *)                       WHEEL_IDX="cu124" ;;  # safe default — old wheels
esac
echo "    Using PyTorch wheel index: $WHEEL_IDX"
uv pip install --pre torch torchvision --index-url "https://download.pytorch.org/whl/$WHEEL_IDX"

echo "==> Verify torch can actually talk to the GPU before doing more setup"
if ! python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"; then
    echo "ERROR: torch.cuda.is_available() returned False."
    echo "Diagnostics:"
    nvidia-smi | head -3 || echo "  nvidia-smi failed"
    ls /dev/nvidia* 2>&1
    python -c "import ctypes; cuda = ctypes.CDLL('libcuda.so.1'); print('cuInit:', cuda.cuInit(0))"
    echo ""
    echo "This pod's CUDA driver isn't initializable. STOP the pod"
    echo "and redeploy with a different template (newer PyTorch image)."
    exit 1
fi

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

echo "==> FA3 dtype patch — required on A100 (Ampere); H100 (Hopper) is fine"
echo "    as-is because it supports the upstream FP8 path. We detect and patch"
echo "    only when needed."
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "    GPU: $GPU_NAME"
if [[ "$GPU_NAME" != *H100* && "$GPU_NAME" != *H200* ]]; then
    FA3_FILE="$WORK/sam3/sam3/perflib/fa3.py"
    if [ -f "$FA3_FILE" ] && grep -q "float8_e4m3fn" "$FA3_FILE"; then
        echo "    Patching $FA3_FILE: float8_e4m3fn -> bfloat16"
        sed -i 's/torch\.float8_e4m3fn/torch.bfloat16/g' "$FA3_FILE"
    fi
fi

echo "==> HuggingFace auth (you'll get prompted to paste a token with"
echo "    'read' access to facebook/sam3 — request access on the model page first)"
hf auth login || huggingface-cli login

echo "==> Download SAM 3 checkpoint (standard non-multiplex video predictor;"
echo "    works on public main code, unlike sam3.1_multiplex)"
cd "$WORK/sam3"
python - <<'PYDL'
# Touch the model builder so the checkpoint gets cached locally now,
# instead of mid-inference (matters because we want clean timings).
from sam3.model_builder import build_sam3_video_predictor
print("Loading SAM 3 video predictor (downloads facebook/sam3 on first run)...")
predictor = build_sam3_video_predictor()
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
