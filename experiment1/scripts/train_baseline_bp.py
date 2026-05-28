"""Vanilla baseline BP — no normalization, no augmentation, no dropout."""
import argparse, json, sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description="Train vanilla BP baseline (no normalization/augmentation/dropout).")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--hidden-layers", default="256,128,64")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)

    print("Loading raw MNIST (NO normalization)...")
    from core.mnist_data import load_mnist_numpy
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=args.data_dir, val_size=args.val_size, normalize=False,
    )
    print(f"Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    hidden = [int(s) for s in args.hidden_layers.split(",")]
    print(f"Vanilla BP: 784 -> {' -> '.join(map(str, hidden))} -> 10")
    print("NO normalization, NO augmentation, NO dropout")

    from core.Net import Net
    model = Net(input_size=784, output_size=10, linears=hidden)
    history = model.train(
        x_train, y_train,
        batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
        use_augmentation=False, dropout_rate=0,
        x_val=x_val, y_val=y_val, metrics_file=args.metrics_path,
    )

    val_acc = model.evaluate(x_val, y_val)
    test_acc = model.evaluate(x_test, y_test)
    model.save_model(args.save_path)

    result = {
        "test_acc": float(test_acc),
        "val_acc": float(val_acc),
        "best_val_acc": float(max(history["val_acc"])),
        "epochs": args.epochs,
        "config": "vanilla_no_norm_no_aug_no_dropout",
    }
    result_path = str(Path(args.save_path).with_suffix(".json"))
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Validation: {val_acc:.4f}, Test: {test_acc:.4f}")
    print(f"Saved: {args.save_path}")
    print(f"Metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
