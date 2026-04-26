import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ImageAugmenter import ImageAugmenter
from MathTools import cross_entropy
from Net import Net


AugmentConfig = Dict[str, Dict[str, object]]


def _base_config() -> AugmentConfig:
    return {
        "translation": {"enabled": False, "max_shift": 2, "probability": 0.0},
        "rotation": {"enabled": False, "max_angle": 12, "probability": 0.0},
        "scaling": {"enabled": False, "scale_range": (0.9, 1.1), "probability": 0.0},
        "flip": {"enabled": False, "horizontal": False, "vertical": False, "probability": 0.0},
        "noise": {"enabled": False, "noise_level": 0.02, "probability": 0.0},
        "erasing": {"enabled": False, "erase_ratio": 0.08, "probability": 0.0},
    }


def build_policy_space() -> Dict[str, AugmentConfig]:
    policies = {}

    policies["clean"] = _base_config()

    cfg = _base_config()
    cfg["translation"] = {"enabled": True, "max_shift": 2, "probability": 1.0}
    policies["shift"] = cfg

    cfg = _base_config()
    cfg["rotation"] = {"enabled": True, "max_angle": 12, "probability": 1.0}
    policies["rotate"] = cfg

    cfg = _base_config()
    cfg["scaling"] = {"enabled": True, "scale_range": (0.85, 1.15), "probability": 1.0}
    policies["scale"] = cfg

    cfg = _base_config()
    cfg["noise"] = {"enabled": True, "noise_level": 0.04, "probability": 1.0}
    policies["noise"] = cfg

    cfg = _base_config()
    cfg["erasing"] = {"enabled": True, "erase_ratio": 0.10, "probability": 1.0}
    policies["erase"] = cfg

    cfg = _base_config()
    cfg["translation"] = {"enabled": True, "max_shift": 2, "probability": 0.8}
    cfg["rotation"] = {"enabled": True, "max_angle": 10, "probability": 0.6}
    cfg["noise"] = {"enabled": True, "noise_level": 0.02, "probability": 0.4}
    policies["mixed_light"] = cfg

    cfg = _base_config()
    cfg["translation"] = {"enabled": True, "max_shift": 3, "probability": 0.8}
    cfg["rotation"] = {"enabled": True, "max_angle": 18, "probability": 0.8}
    cfg["scaling"] = {"enabled": True, "scale_range": (0.8, 1.2), "probability": 0.6}
    cfg["erasing"] = {"enabled": True, "erase_ratio": 0.12, "probability": 0.4}
    policies["mixed_strong"] = cfg

    return policies


@dataclass
class EpsilonGreedyAugmentationAgent:
    action_names: List[str]
    epsilon: float = 0.25
    decay: float = 0.96
    min_epsilon: float = 0.05
    random_seed: int = 7
    counts: Dict[str, int] = field(init=False)
    values: Dict[str, float] = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.random_seed)
        self.counts = {name: 0 for name in self.action_names}
        self.values = {name: 0.0 for name in self.action_names}

    def choose(self) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_names).item()
        return max(self.action_names, key=lambda name: self.values[name])

    def update(self, action: str, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.values[action] += (reward - self.values[action]) / n
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

    def snapshot(self) -> Dict[str, object]:
        return {
            "epsilon": self.epsilon,
            "counts": dict(self.counts),
            "values": dict(self.values),
            "best_action": max(self.action_names, key=lambda name: self.values[name]),
        }


def _accuracy(model: Net, x: np.ndarray, y: np.ndarray) -> float:
    probs = model.predict(x)
    return float(np.mean(np.argmax(probs, axis=1) == np.argmax(y, axis=1)))


def train_bp_with_rl_augmentation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_layers: List[int] = None,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
    save_path: str = "mnist_model.npy",
) -> Tuple[Net, List[Dict[str, object]]]:
    """Train the NumPy BP model while a bandit agent learns augmentation policy choice."""

    model = Net(input_size=784, output_size=10, linears=hidden_layers or [128, 64])
    augmenter = ImageAugmenter(image_shape=(28, 28), random_seed=42)
    policies = build_policy_space()
    agent = EpsilonGreedyAugmentationAgent(list(policies.keys()))
    history = []
    best_val_acc = _accuracy(model, x_val, y_val)

    for epoch in range(epochs):
        action = agent.choose()
        config = policies[action]
        order = np.arange(len(x_train))
        np.random.shuffle(order)
        epoch_losses = []

        for start in range(0, len(x_train), batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            if action != "clean":
                xb = augmenter.augment_batch(xb, config=config)

            probs = model.forward(xb, training=True)
            model.backward(yb, lr=lr)
            epoch_losses.append(cross_entropy(yb, probs))

        val_acc = _accuracy(model, x_val, y_val)
        reward = val_acc - best_val_acc
        best_val_acc = max(best_val_acc, val_acc)
        agent.update(action, reward)

        row = {
            "epoch": epoch + 1,
            "action": action,
            "avg_loss": float(np.mean(epoch_losses)),
            "val_acc": val_acc,
            "reward": float(reward),
            "agent": agent.snapshot(),
        }
        history.append(row)
        print(
            f"Epoch {row['epoch']:02d} | action={action:<12} "
            f"| loss={row['avg_loss']:.4f} | val_acc={val_acc:.4f} | reward={reward:+.4f}"
        )

    model.save_model(save_path)
    with open("rl_augmentation_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return model, history

