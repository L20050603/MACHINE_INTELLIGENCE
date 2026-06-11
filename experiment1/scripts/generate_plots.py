"""Train all models with metrics tracking and generate comparison plots."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

ARTIFACTS = PROJECT_DIR / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

# ──────────────────────────────────────────────────────
# 1. Pure BP: loss + val_acc curves
# ──────────────────────────────────────────────────────

def plot_bp_metrics():
    from core.Net import Net
    from core.mnist_data import load_mnist_numpy

    print("=" * 50)
    print("Training: Pure BP (MLP Baseline)")
    print("=" * 50)

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=str(PROJECT_DIR / "data"), val_size=5000
    )

    model = Net(input_size=784, output_size=10, linears=[256, 128, 64])
    metrics_file = str(PLOTS_DIR / "bp_metrics.json")
    history = model.train(
        x_train, y_train,
        batch_size=64, epochs=15, lr=0.001,
        lr_decay_step=10, lr_decay_gamma=0.5,
        x_val=x_val, y_val=y_val,
        metrics_file=metrics_file,
    )

    test_acc = model.evaluate(x_test, y_test)
    model.save_model(str(ARTIFACTS / "models" / "enhanced_bp.npy"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=5, linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("纯 BP — 训练 Loss 曲线")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [v * 100 for v in history["val_acc"]], "r-o", markersize=5, linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"纯 BP — 验证准确率 (Test: {test_acc * 100:.2f}%)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "bp_loss_valacc.png")
    plt.close(fig)
    print(f"  -> saved: bp_loss_valacc.png (Test acc: {test_acc:.4f})")
    return test_acc


# ──────────────────────────────────────────────────────
# 2. RL-enhanced BP: loss + val_acc + reward curves
# ──────────────────────────────────────────────────────

def plot_rl_metrics():
    from core.mnist_data import load_mnist_numpy
    from core.rl_augmenter import train_bp_with_rl_augmentation

    print("=" * 50)
    print("Training: RL-enhanced BP")
    print("=" * 50)

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_numpy(
        data_dir=str(PROJECT_DIR / "data"), val_size=5000
    )

    history_path = str(PLOTS_DIR / "rl_history.json")
    model, history = train_bp_with_rl_augmentation(
        x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val,
        hidden_layers=[256, 128, 64],
        epochs=10, batch_size=64, lr=0.001,
        save_path=str(ARTIFACTS / "models" / "rl_enhanced_bp.npy"),
        history_path=history_path,
    )
    test_acc = model.evaluate(x_test, y_test)

    epochs = [r["epoch"] for r in history]
    losses = [r["avg_loss"] for r in history]
    val_accs = [r["val_acc"] * 100 for r in history]
    rewards = [r["reward"] for r in history]
    actions = [r["action"] for r in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    axes[0].plot(epochs, losses, "b-o", markersize=5, linewidth=1.8)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("RL 增强 BP — 训练 Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_accs, "g-o", markersize=5, linewidth=1.8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"RL 增强 BP — 验证准确率 (Test: {test_acc * 100:.2f}%)")
    axes[1].grid(True, alpha=0.3)

    colors = ["g" if r > 0 else "r" for r in rewards]
    axes[2].bar(epochs, rewards, color=colors, alpha=0.7)
    # Annotate with action names
    for i, (ep, r, a) in enumerate(zip(epochs, rewards, actions)):
        axes[2].annotate(a, (ep, r), textcoords="offset points", xytext=(0, 8 if r >= 0 else -14),
                         ha="center", fontsize=7, rotation=45)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("RL 增强 BP — Reward (策略选择)")
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rl_loss_valacc_reward.png")
    plt.close(fig)
    print(f"  -> saved: rl_loss_valacc_reward.png (Test acc: {test_acc:.4f})")
    return test_acc


# ──────────────────────────────────────────────────────
# 3. ACGAN: G/D loss curves
# ──────────────────────────────────────────────────────

def plot_acgan_metrics():
    from core.acgan_synthesizer import train_acgan

    print("=" * 50)
    print("Training: ACGAN (for G/D loss curves)")
    print("=" * 50)

    metrics_file = str(PLOTS_DIR / "acgan_metrics.json")

    # Check if ACGAN already has metrics saved
    acgan_checkpoint = ARTIFACTS / "generated" / "acgan_mnist.pth"

    if metrics_file and Path(metrics_file).exists():
        # Load existing metrics from a previous ACGAN training run
        # But we need fresh ones. Let's check the checkpoint.
        pass

    # Train ACGAN for curves, save checkpoint separately to avoid overwriting prod model
    generator, hist = train_acgan(
        data_dir=str(PROJECT_DIR / "data"),
        output_dir=str(PLOTS_DIR),
        epochs=50,
        batch_size=128,
        metrics_file=metrics_file,
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    epochs = range(1, len(hist["g_loss"]) + 1)

    ax.plot(epochs, hist["g_loss"], "b-", linewidth=1.5, alpha=0.8, label="Generator Loss")
    ax.plot(epochs, hist["d_loss"], "r-", linewidth=1.5, alpha=0.8, label="Discriminator Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("ACGAN — Generator / Discriminator Loss 曲线")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "acgan_gd_loss.png")
    plt.close(fig)
    print("  -> saved: acgan_gd_loss.png")
    return None


# ──────────────────────────────────────────────────────
# 4. CNN / Residual BP: loss + val_acc curves
# ──────────────────────────────────────────────────────

def plot_cnn_metrics():
    from core.Net2 import MNISTExpert
    from core.mnist_downloader import ensure_mnist_available

    print("=" * 50)
    print("Training: CNN / Residual BP")
    print("=" * 50)

    data_dir = str(PROJECT_DIR / "data")
    ensure_mnist_available(data_dir)

    expert = MNISTExpert(lr=0.001, data_dir=data_dir)
    metrics_file = str(PLOTS_DIR / "cnn_metrics.json")
    history = expert.train_model(epochs=15, batch_size=64, metrics_file=metrics_file)
    test_acc = expert.evaluate()
    expert.save(str(ARTIFACTS / "models" / "cnn_residual.pth"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=5, linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("CNN — 训练 Loss 曲线")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [v * 100 for v in history["val_acc"]], "r-o", markersize=5, linewidth=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"CNN — 验证准确率 (Test: {test_acc:.2f}%)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cnn_loss_valacc.png")
    plt.close(fig)
    print(f"  -> saved: cnn_loss_valacc.png (Test acc: {test_acc:.2f}%)")
    return test_acc / 100.0


# ──────────────────────────────────────────────────────
# 5. Four-model accuracy comparison
# ──────────────────────────────────────────────────────

def plot_accuracy_comparison(accuracies):
    """accuracies: dict of model_name -> accuracy (0-1 float)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(accuracies.keys())
    values = [accuracies[n] * 100 for n in names]
    colors = ["#315f9e", "#d4552d", "#0b7a75", "#7b4ea3"]

    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("四模型 MNIST 测试准确率对比")
    ax.set_ylim(min(values) - 2, max(values) + 3)
    ax.grid(True, alpha=0.2, axis="y")
    ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_comparison.png")
    plt.close(fig)
    print("  -> saved: accuracy_comparison.png")


# ──────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────

def main():
    print(f"Plots will be saved to: {PLOTS_DIR}")
    print()

    accuracies = {}

    # 1. Pure BP
    acc = plot_bp_metrics()
    accuracies["增强版BP"] = acc

    # 2. RL-enhanced BP
    acc = plot_rl_metrics()
    accuracies["强化学习增强版BP"] = acc

    # 3. ACGAN G/D loss (run separately, not used for accuracy)
    plot_acgan_metrics()

    # 4. CNN
    acc = plot_cnn_metrics()
    accuracies["CNN Residual"] = acc

    # 5. GAN-augmented BP (quick eval of existing model)
    print("=" * 50)
    print("Evaluating: GAN-augmented BP")
    from core.Net import Net
    from core.mnist_data import load_mnist_numpy
    _, _, _, _, x_test, y_test = load_mnist_numpy(
        data_dir=str(PROJECT_DIR / "data"), val_size=5000
    )
    gan_path = ARTIFACTS / "models" / "acgan_enhanced_bp.npy"
    if gan_path.exists():
        gan_model = Net(input_size=784, output_size=10, linears=[256, 128, 64])
        gan_model.load_model(str(gan_path))
        gan_acc = gan_model.evaluate(x_test, y_test)
        accuracies["ACGAN增强版BP"] = gan_acc
        print(f"  ACGAN-enhanced BP test accuracy: {gan_acc:.4f}")
    else:
        print("  WARNING: acgan_enhanced_bp.npy not found, skipping ACGAN accuracy")

    # 6. Accuracy comparison chart
    print()
    print("=" * 50)
    print("Generating accuracy comparison chart...")
    plot_accuracy_comparison(accuracies)

    # Print summary
    print()
    print("=" * 50)
    print("Summary: Test Accuracies")
    print("=" * 50)
    for name, acc in accuracies.items():
        print(f"  {name:<22s}: {acc * 100:.2f}%")
    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
