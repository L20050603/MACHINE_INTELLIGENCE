import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from Net import Net


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
MODEL_CANDIDATES = [
    ROOT / "mnist_model.npy",
    ROOT.parent / "mnist_model.npy",
]


def _load_model():
    model = Net(input_size=784, output_size=10, linears=[128, 64])
    for path in MODEL_CANDIDATES:
        if path.exists():
            model.load_model(str(path))
            return model, str(path), True
    return model, "", False


MODEL, MODEL_PATH, MODEL_READY = _load_model()


class MnistHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._json(
                {
                    "ok": True,
                    "modelReady": MODEL_READY,
                    "modelPath": MODEL_PATH,
                    "message": "BP model loaded" if MODEL_READY else "No trained model file found",
                }
            )
            return
        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            pixels = np.asarray(payload.get("pixels", []), dtype=np.float32)
            if pixels.size != 784:
                raise ValueError("pixels must contain exactly 784 numbers")
            pixels = np.clip(pixels.reshape(1, 784), 0.0, 1.0)
            probs = MODEL.predict(pixels)[0]
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
                    "modelReady": MODEL_READY,
                }
            )
        except Exception as exc:
            self._json({"error": str(exc)}, status=400)

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
    print("Place a trained mnist_model.npy in experiment1/ or project root for real predictions.")
    server.serve_forever()


if __name__ == "__main__":
    run()

