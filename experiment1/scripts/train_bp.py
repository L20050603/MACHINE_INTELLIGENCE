import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core.Net import Net
from core.mnist_data import load_mnist_numpy


def parse_hidden_layers(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the baseline NumPy BP/MLP model.")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "artifacts" / "models" / "mlp_baseline.npy"))
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--hidden-layers", default="256,128,64")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-decay-step", type=int, default=0)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading MNIST with torchvision...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=args.data_dir,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        val_size=args.val_size,
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    print(f"Network: 784 -> {' -> '.join(map(str, hidden_layers))} -> 10")
    model = Net(input_size=784, output_size=10, linears=hidden_layers)
    model.train(
        x_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        counts=2000,
        use_augmentation=not args.no_augmentation,
        dropout_rate=args.dropout,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma,
    )

    val_acc = model.evaluate(x_val, y_val)
    test_acc = model.evaluate(x_test, y_test)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    model.save_model(args.save_path)


if __name__ == "__main__":
    main()
