import json

import numpy as np

from .ImageAugmenter import ImageAugmenter
from .MathTools import ReLU, ReLU_derivative, cross_entropy, softmax


class Net:
    def __init__(self, input_size, output_size, linears=None):
        dims = [input_size] + (linears or []) + [output_size]
        self.layers = []
        self.cache = []
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])
            self.layers.append(
                {
                    "W": np.random.randn(dims[i], dims[i + 1]) * std,
                    "b": np.zeros((1, dims[i + 1])),
                    "mW": np.zeros((dims[i], dims[i + 1])),
                    "vW": np.zeros((dims[i], dims[i + 1])),
                    "mb": np.zeros((1, dims[i + 1])),
                    "vb": np.zeros((1, dims[i + 1])),
                }
            )
        self.t = 0

    def forward(self, X, training=True, dropout_rate=0.1):
        A = X
        self.cache = [{"A": X, "Z": None}]
        for i, layer in enumerate(self.layers):
            Z = A @ layer["W"] + layer["b"]
            if i == len(self.layers) - 1:
                A = softmax(Z)
            else:
                A = ReLU(Z)
                if training and dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) > dropout_rate) / (1 - dropout_rate)
                    A *= mask
                else:
                    mask = None
            self.cache.append({"A": A, "Z": Z, "mask": mask if i < len(self.layers) - 1 else None})
        return A

    def backward(self, y_true, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer="adam"):
        m = y_true.shape[0]
        self.t += 1
        dz = self.cache[-1]["A"] - y_true
        use_sgd = (optimizer == "sgd")

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self.cache[i]["A"]
            dW = (A_prev.T @ dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz_prev = (dz @ layer["W"].T) * ReLU_derivative(self.cache[i]["Z"])
                mask = self.cache[i].get("mask")
                if mask is not None:
                    dz_prev *= mask
            else:
                dz_prev = None

            if use_sgd:
                layer["W"] -= lr * dW
                layer["b"] -= lr * db
            else:
                layer["mW"] = beta1 * layer["mW"] + (1 - beta1) * dW
                layer["mb"] = beta1 * layer["mb"] + (1 - beta1) * db
                layer["vW"] = beta2 * layer["vW"] + (1 - beta2) * (dW ** 2)
                layer["vb"] = beta2 * layer["vb"] + (1 - beta2) * (db ** 2)

                mW_hat = layer["mW"] / (1 - beta1 ** self.t)
                mb_hat = layer["mb"] / (1 - beta1 ** self.t)
                vW_hat = layer["vW"] / (1 - beta2 ** self.t)
                vb_hat = layer["vb"] / (1 - beta2 ** self.t)

                layer["W"] -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
                layer["b"] -= lr * mb_hat / (np.sqrt(vb_hat) + epsilon)

            if dz_prev is not None:
                dz = dz_prev

    def train(
        self,
        x_train,
        y_train,
        batch_size,
        epochs,
        lr,
        counts=100,
        use_augmentation=True,
        dropout_rate=0.1,
        lr_decay_step=0,
        lr_decay_gamma=0.5,
        x_val=None,
        y_val=None,
        metrics_file=None,
        optimizer="adam",
    ):
        n_samples = len(x_train)
        current_lr = lr
        augmenter = ImageAugmenter(image_shape=(28, 28), random_seed=42)
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            if lr_decay_step and epoch > 0 and epoch % lr_decay_step == 0:
                current_lr *= lr_decay_gamma
                print(f"--- Learning rate decayed to: {current_lr:.6f} ---")

            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            running_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                xb = x_train[batch_indices]
                yb = y_train[batch_indices]
                if use_augmentation and np.random.rand() > 0.5:
                    xb = augmenter.augment_batch(xb)

                y_pred = self.forward(xb, training=True, dropout_rate=dropout_rate)
                self.backward(yb, current_lr, optimizer=optimizer)
                running_loss += cross_entropy(yb, y_pred)
                n_batches += 1

                if i % counts == 0:
                    avg_loss = running_loss / n_batches
                    print(f"Epoch:{epoch} | Batch:{i // batch_size} | Avg Loss:{avg_loss:.4f}")

            epoch_loss = running_loss / n_batches
            history["train_loss"].append(float(epoch_loss))

            if x_val is not None and y_val is not None:
                val_acc = self.evaluate(x_val, y_val)
                history["val_acc"].append(float(val_acc))
                print(f"Epoch:{epoch} | Train Loss:{epoch_loss:.4f} | Val Acc:{val_acc:.4f}")

            if metrics_file:
                import json as _json
                with open(metrics_file, "w") as f:
                    _json.dump(history, f)

        return history

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.forward(x, training=False)

    def save_model(self, filepath):
        dims = [self.layers[0]["W"].shape[0]] + [l["W"].shape[1] for l in self.layers]
        model_data = {"linears": dims[1:-1], "layers": self.layers, "t": self.t}
        try:
            np.save(filepath, model_data)
            print(f"模型已成功保存到 {filepath}")
        except Exception as e:
            print(f"保存模型失败: {e}")

    def load_model(self, filepath):
        try:
            model_data = np.load(filepath, allow_pickle=True).item()
            saved_layers = model_data["layers"]
            dims = [saved_layers[0]["W"].shape[0]] + [l["W"].shape[1] for l in saved_layers]
            self.layers = saved_layers
            self.t = model_data["t"]
            print(f"模型已成功从 {filepath} 加载")
            print(f"网络层数: {len(self.layers)}, 架构: {' -> '.join(map(str, dims))}")
            print(f"Adam迭代次数: {self.t}")
        except FileNotFoundError:
            print(f"模型文件不存在: {filepath}")
        except Exception as e:
            print(f"加载模型失败: {e}")

    def evaluate(self, x_test, y_test):
        y_pred = self.forward(x_test, training=False)
        predictions = np.argmax(y_pred, axis=-1)
        labels = np.argmax(y_test, axis=-1)
        return np.mean(predictions == labels)
