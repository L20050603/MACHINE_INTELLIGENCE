import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from core.acgan_synthesizer import generate_numpy_samples, train_acgan
from core.mnist_data import load_mnist_numpy, to_one_hot
from core.Net import Net


def parse_args():
    parser = argparse.ArgumentParser(description="Train BP with ACGAN-generated MNIST samples.")
    parser.add_argument("--data-dir", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--gan-dir", default=str(PROJECT_DIR / "artifacts" / "generated"))
    parser.add_argument("--gan-epochs", type=int, default=3)
    parser.add_argument("--bp-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--synthetic-per-class", type=int, default=300)
    parser.add_argument("--synthetic-min-confidence", type=float, default=0.0)
    parser.add_argument("--retrain-gan", action="store_true")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--save-path", default=str(PROJECT_DIR / "artifacts" / "models" / "mlp_gan.npy"))
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.gan_dir).mkdir(parents=True, exist_ok=True)
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = os.path.join(args.gan_dir, "acgan_mnist.pth")
    if args.retrain_gan and os.path.exists(checkpoint):
        os.remove(checkpoint)

    if not os.path.exists(checkpoint):
        print("No ACGAN checkpoint found. Training generator first...")
        train_acgan(
            data_dir=args.data_dir,
            output_dir=args.gan_dir,
            epochs=args.gan_epochs,
            batch_size=128,
        )

    print("Loading real MNIST...")
    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=args.data_dir,
        limit_train=args.limit_train,
        val_size=5000,
    )

    print("Generating synthetic samples...")
    images_by_class, labels_by_class = generate_numpy_samples(
        checkpoint,
        per_class=args.synthetic_per_class,
        min_class_confidence=args.synthetic_min_confidence,
    )
    x_synth = np.vstack(images_by_class).astype(np.float32)
    y_synth = to_one_hot(np.concatenate(labels_by_class))

    x_mix = np.vstack([x_train, x_synth])
    y_mix = np.vstack([y_train, y_synth])
    order = np.arange(len(x_mix))
    np.random.shuffle(order)
    x_mix = x_mix[order]
    y_mix = y_mix[order]

    print(f"Real train: {x_train.shape}, Synthetic: {x_synth.shape}, Mixed: {x_mix.shape}")
    model = Net(input_size=784, output_size=10, linears=[128, 64])
    model.train(
        x_mix,
        y_mix,
        batch_size=args.batch_size,
        epochs=args.bp_epochs,
        lr=args.lr,
        counts=2000,
    )

    val_acc = model.evaluate(x_val, y_val)
    test_acc = model.evaluate(x_test, y_test)
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    model.save_model(args.save_path)


if __name__ == "__main__":
    main()
