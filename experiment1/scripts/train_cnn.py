import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core.Net2 import MNISTExpert
from core.mnist_downloader import ensure_mnist_available


def parse_args():
    parser = argparse.ArgumentParser(description="Train the PyTorch CNN/Residual model.")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "artifacts" / "models" / "cnn_residual.pth"))
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    ensure_mnist_available(args.data_dir)

    expert = MNISTExpert(lr=args.lr, data_dir=args.data_dir)
    expert.train_model(epochs=args.epochs, batch_size=args.batch_size)
    expert.evaluate()
    expert.save(args.save_path)
    print(f"Saved CNN model to: {Path(args.save_path).resolve()}")


if __name__ == "__main__":
    main()
