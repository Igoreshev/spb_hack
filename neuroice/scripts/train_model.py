import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from solution import train_and_save

if __name__ == "__main__":
    train_root = root.parent / "train"
    manifest = train_root / "train.csv"
    audio_dir = train_root / "train"
    train_and_save(
        manifest_path=manifest,
        audio_root=audio_dir,
        model_path=root / "model.pkl",
        scaler_path=root / "scaler.pkl",
    )
