"""Fine-tune YOLO26 on the Roboflow football-players-detection dataset.

Run this once after downloading the dataset (see README). Training takes
~30-60 min on an RTX 4070 Laptop with the defaults below; longer if you
bump model_size or epochs.

The Roboflow data.yaml ships with weird relative paths (../train/images)
that don't resolve when ultralytics tries to read them. We re-emit a
clean copy at datasets/data_soccer.yaml with absolute paths before
training, so we don't mutate the original.

After training, the best weights land at:
    runs/ball_finetune_v1/weights/best.pt

Use them with:
    python track_ball.py \\
        --input clips/test.mp4 \\
        --output out/finetuned.mp4 \\
        --overlay ring \\
        --model runs/ball_finetune_v1/weights/best.pt \\
        --ball-class 0
"""
from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "datasets" / "football-players-detection-17"
CLEAN_YAML = PROJECT_ROOT / "datasets" / "data_soccer.yaml"


def emit_clean_yaml() -> Path:
    """Write a clean data.yaml with absolute paths so ultralytics finds
    the images regardless of cwd."""
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_ROOT}. "
            f"Run datasets/download.py first."
        )
    config = {
        "path": str(DATASET_ROOT),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 4,
        "names": ["ball", "goalkeeper", "player", "referee"],
    }
    CLEAN_YAML.parent.mkdir(parents=True, exist_ok=True)
    CLEAN_YAML.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Wrote clean dataset config: {CLEAN_YAML}")
    return CLEAN_YAML


def main() -> None:
    data_yaml = emit_clean_yaml()

    # Start from the pretrained YOLO26-s detection model. -s (small) is a
    # good balance for the 4070 Laptop: better than -n for small objects,
    # fits in 8 GB VRAM at imgsz=1280, trains in ~30-60 min for 100 epochs.
    model = YOLO("yolo26s.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        # imgsz=640 is what Roboflow's yolo11 preset originally resized
        # source images to, so we're not throwing away information. batch=4
        # comfortably fits the 4070 Laptop's 8 GB VRAM at this resolution.
        # multi_scale stays OFF — at 1.5x scale it tries imgsz=1696 which
        # OOMs. Inference can still use imgsz=1280 because Ultralytics
        # supports dynamic input sizes at test time.
        imgsz=640,
        batch=4,
        device=0,              # GPU 0
        patience=20,           # early-stop if val mAP plateaus 20 epochs
        cache=True,            # cache images in RAM (we have 16 GB system RAM)
        amp=True,              # mixed precision — halves activation memory
        project="runs",
        name="ball_finetune_v1",
        # ball is 1/4 of classes; the inference pipeline filters to ball-only
        # via --ball-class. Training on all classes is fine and actually helps
        # the model learn "what is NOT a ball" (players, refs, goalkeepers).
    )

    # Quick post-training validation summary on the held-out test split.
    print("\n--- final evaluation on test split ---")
    metrics = model.val(data=str(data_yaml), split="test", imgsz=1280)
    print(metrics.results_dict)
    print("\nFine-tuned weights at: runs/ball_finetune_v1/weights/best.pt")


if __name__ == "__main__":
    main()
