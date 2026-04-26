import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core.mnist_data import load_mnist_numpy
from core.rl_augmenter import train_bp_with_rl_augmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Train the NumPy BP model with RL-selected augmentation policies.")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "artifacts" / "models" / "mlp_rl.npy"))
    parser.add_argument("--history-path", default=str(PROJECT_DIR / "artifacts" / "rl_augmentation_history.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.history_path).parent.mkdir(parents=True, exist_ok=True)
    print("Loading MNIST with torchvision...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=args.data_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        val_size=args.val_size,
    )

    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")
    model, history = train_bp_with_rl_augmentation(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.save_path,
        history_path=args.history_path,
    )

    test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Saved model to: {Path(args.save_path).resolve()}")
    print(f"Saved RL policy history to: {Path(args.history_path).resolve()}")
    print(f"Last selected best policy: {history[-1]['agent']['best_action']}")


if __name__ == "__main__":
    main()
