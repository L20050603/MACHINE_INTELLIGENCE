import base64
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from core.Net import Net


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
ARTIFACTS_ROOT = ROOT / "artifacts"
MODEL_ROOT = ARTIFACTS_ROOT / "models"

MODEL_REGISTRY = {
    "bp": {
        "name": "BP",
        "type": "mlp",
        "path": MODEL_ROOT / "bp.npy",
        "note": "原始反向传播，无归一化，无增强",
        "normalize": False,
    },
    "enhanced_bp": {
        "name": "增强版BP",
        "type": "mlp",
        "path": MODEL_ROOT / "enhanced_bp.npy",
        "note": "归一化 + Dropout + Adam + LR衰减",
        "normalize": True,
    },
    "rl_enhanced_bp": {
        "name": "强化学习增强版BP",
        "type": "mlp",
        "path": MODEL_ROOT / "rl_enhanced_bp.npy",
        "note": "epsilon-greedy 选择增强策略",
        "normalize": True,
    },
    "acgan_enhanced_bp": {
        "name": "ACGAN增强版BP",
        "type": "mlp",
        "path": MODEL_ROOT / "acgan_enhanced_bp.npy",
        "note": "真实样本 + ACGAN 合成样本",
        "normalize": True,
    },
    "cnn_residual": {
        "name": "CNN/Residual BP",
        "type": "cnn",
        "path": MODEL_ROOT / "cnn_residual.pth",
        "note": "PyTorch CNN，仍由反向传播训练",
    },
}

LEGACY_MODEL_PATH = ARTIFACTS_ROOT / "mnist_model.npy"
GENERATOR_CANDIDATES = [
    ARTIFACTS_ROOT / "acgan" / "acgan_mnist.pth",
    ARTIFACTS_ROOT / "generated" / "acgan_mnist.pth",
    ROOT / "generated" / "acgan_mnist.pth",
]

MODEL_CACHE = {}


def _find_generator():
    for path in GENERATOR_CANDIDATES:
        if path.exists():
            return str(path), True
    return "", False


def _available_models():
    models = []
    for model_id, meta in MODEL_REGISTRY.items():
        models.append(
            {
                "id": model_id,
                "name": meta["name"],
                "type": meta["type"],
                "note": meta["note"],
                "ready": meta["path"].exists(),
                "path": str(meta["path"]) if meta["path"].exists() else "",
            }
        )
    if LEGACY_MODEL_PATH.exists():
        models.append(
            {
                "id": "legacy",
                "name": "当前默认权重",
                "type": "mlp",
                "note": "兼容旧路径 artifacts/mnist_model.npy",
                "ready": True,
                "path": str(LEGACY_MODEL_PATH),
            }
        )
    return models


def _default_model_id():
    for model_id in ("bp", "enhanced_bp", "rl_enhanced_bp", "acgan_enhanced_bp", "cnn_residual", "legacy"):
        if _model_path(model_id).exists():
            return model_id
    return "bp"


def _model_path(model_id):
    if model_id == "legacy":
        return LEGACY_MODEL_PATH
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model id: {model_id}")
    return MODEL_REGISTRY[model_id]["path"]


def _model_type(model_id):
    if model_id == "legacy":
        return "mlp"
    return MODEL_REGISTRY[model_id]["type"]


def _load_mlp(path):
    cache_key = ("mlp", str(path))
    if cache_key not in MODEL_CACHE:
        # Auto-detect architecture from saved file
        saved = np.load(str(path), allow_pickle=True).item()
        layer_dims = [saved["layers"][0]["W"].shape[0]] + [l["W"].shape[1] for l in saved["layers"]]
        linears = layer_dims[1:-1]  # exclude input (784) and output (10)
        model = Net(input_size=784, output_size=10, linears=linears)
        model.load_model(str(path))
        MODEL_CACHE[cache_key] = model
    return MODEL_CACHE[cache_key]


def _load_cnn(path):
    cache_key = ("cnn", str(path))
    if cache_key not in MODEL_CACHE:
        import torch
        from core.Net2 import ConvNet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ConvNet().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        MODEL_CACHE[cache_key] = (model, device)
    return MODEL_CACHE[cache_key]


def _shift_image(image, dy, dx):
    result = np.zeros_like(image)
    src_y0 = max(0, -dy)
    src_y1 = min(image.shape[0], image.shape[0] - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(image.shape[1], image.shape[1] - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(image.shape[0], image.shape[0] + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(image.shape[1], image.shape[1] + dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return result


def _shear_x(image, shear):
    result = np.zeros_like(image)
    center_y = (image.shape[0] - 1) / 2.0
    for y in range(image.shape[0]):
        offset = shear * (y - center_y)
        for x in range(image.shape[1]):
            value = image[y, x]
            if value <= 0:
                continue
            nx = int(round(x + offset))
            if 0 <= nx < image.shape[1]:
                result[y, nx] = max(result[y, nx], value)
    return result


def _variants(pixels):
    image = pixels.reshape(28, 28)
    return [
        image,
        _shift_image(image, 0, -1),
        _shift_image(image, 0, 1),
        _shift_image(image, -1, 0),
        _shift_image(image, 1, 0),
        _shear_x(image, -0.18),
        _shear_x(image, -0.28),
        _shear_x(image, 0.14),
    ]


def _pool_probs(probs):
    pooled = 0.7 * probs.mean(axis=0) + 0.3 * probs.max(axis=0)
    return pooled / np.sum(pooled)


def _predict_mlp(path, pixels, normalize=True):
    model = _load_mlp(path)
    batch = np.stack([np.clip(v.reshape(784), 0.0, 1.0) for v in _variants(pixels)]).astype(np.float32)
    if normalize:
        batch = (batch - 0.1307) / 0.3081
    return _pool_probs(model.predict(batch))


def _predict_cnn(path, pixels):
    import torch
    import torch.nn.functional as F

    model, device = _load_cnn(path)
    batch = np.stack([np.clip(v, 0.0, 1.0) for v in _variants(pixels)]).astype(np.float32)
    tensor = torch.from_numpy(batch[:, None, :, :]).to(device)
    tensor = (tensor - 0.1307) / 0.3081
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()
    return _pool_probs(probs)


def _predict(model_id, pixels):
    path = _model_path(model_id)
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found: {path}")
    normalize = MODEL_REGISTRY.get(model_id, {}).get("normalize", True)
    if _model_type(model_id) == "cnn":
        return _predict_cnn(path, pixels)
    return _predict_mlp(path, pixels, normalize=normalize)


GENERATOR_PATH, GENERATOR_READY = _find_generator()


class MnistHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            models = _available_models()
            self._json(
                {
                    "ok": True,
                    "models": models,
                    "defaultModel": _default_model_id(),
                    "modelReady": any(model["ready"] for model in models),
                    "generatorReady": GENERATOR_READY,
                    "generatorPath": GENERATOR_PATH,
                }
            )
            return
        if parsed.path == "/api/models":
            self._json({"models": _available_models(), "defaultModel": _default_model_id()})
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict":
            self._handle_predict()
            return
        if parsed.path == "/api/generate":
            self._handle_generate()
            return
        self.send_error(404)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _handle_predict(self):
        try:
            payload = self._read_json()
            model_id = payload.get("modelId") or _default_model_id()
            pixels = np.asarray(payload.get("pixels", []), dtype=np.float32)
            if pixels.size != 784:
                raise ValueError("pixels must contain exactly 784 numbers")
            pixels = np.clip(pixels.reshape(784), 0.0, 1.0)
            probs = _predict(model_id, pixels)
            pred = int(np.argmax(probs))
            top3 = [
                {"label": int(i), "probability": float(probs[i])}
                for i in np.argsort(probs)[::-1][:3]
            ]
            self._json(
                {
                    "prediction": pred,
                    "confidence": float(probs[pred]),
                    "probabilities": [float(x) for x in probs],
                    "top3": top3,
                    "modelId": model_id,
                    "modelReady": True,
                    "tta": True,
                }
            )
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)

    def _handle_generate(self):
        try:
            if not GENERATOR_READY:
                raise FileNotFoundError("No trained ACGAN generator found")
            payload = self._read_json()
            digit = int(payload.get("digit", 0))
            count = int(payload.get("count", 6))
            if digit < 0 or digit > 9:
                raise ValueError("digit must be between 0 and 9")
            count = max(1, min(count, 8))
            from core.acgan_synthesizer import generate_digit_png

            png = generate_digit_png(GENERATOR_PATH, digit=digit, count=count)
            encoded = base64.b64encode(png).decode("ascii")
            self._json(
                {
                    "digit": digit,
                    "count": count,
                    "image": f"data:image/png;base64,{encoded}",
                    "generatorReady": True,
                }
            )
        except Exception as exc:
            self._json({"error": str(exc), "generatorReady": GENERATOR_READY}, status=400)

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host="127.0.0.1", port=8000):
    os.chdir(WEB_ROOT)
    server = ThreadingHTTPServer((host, port), MnistHandler)
    print(f"MNIST BP web app: http://{host}:{port}")
    for model in _available_models():
        state = "ready" if model["ready"] else "missing"
        print(f"{state:7} {model['id']}: {model['path'] or model['name']}")
    print(f"ACGAN generator: {GENERATOR_PATH or 'not found'}")
    server.serve_forever()


if __name__ == "__main__":
    run()
