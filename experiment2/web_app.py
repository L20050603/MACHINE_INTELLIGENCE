from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

try:
    from eight_puzzle import HEURISTICS, SearchResult, parse_state, solve_astar
except ModuleNotFoundError:
    from experiment2.eight_puzzle import HEURISTICS, SearchResult, parse_state, solve_astar


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
HOST = "127.0.0.1"
PORT = 8002


def state_to_rows(state):
    return [list(state[i : i + 3]) for i in range(0, 9, 3)]


def result_to_json(result: SearchResult):
    return {
        "found": result.found,
        "message": result.message,
        "heuristic": result.heuristic,
        "weight": result.weight,
        "depth": result.depth,
        "moves": result.moves,
        "path": [state_to_rows(state) for state in result.path],
        "expanded": result.expanded,
        "generated": result.generated,
        "max_frontier": result.max_frontier,
        "elapsed": result.elapsed,
        "start": state_to_rows(result.start),
        "goal": state_to_rows(result.goal),
    }


class EightPuzzleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/heuristics":
            self.send_json({"heuristics": list(HEURISTICS) + ["pattern_db"]})
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            data = self.read_json_body()
            start = parse_state(data.get("start", ""))
            goal = parse_state(data.get("goal", ""))
            heuristic = data.get("heuristic", "pattern_db")
            weight = float(data.get("weight", 1.0))
            max_expanded = int(data.get("max_expanded", 200000))

            if path == "/api/solve":
                result = solve_astar(
                    start=start,
                    goal=goal,
                    heuristic=heuristic,
                    weight=weight,
                    max_expanded=max_expanded,
                )
                self.send_json(result_to_json(result))
                return

            if path == "/api/compare":
                rows = []
                for name in list(HEURISTICS) + ["pattern_db"]:
                    result = solve_astar(
                        start=start,
                        goal=goal,
                        heuristic=name,
                        weight=weight,
                        max_expanded=max_expanded,
                    )
                    rows.append(result_to_json(result))
                self.send_json({"rows": rows})
                return

            self.send_json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)


def main():
    WEB_ROOT.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), EightPuzzleHandler)
    print(f"Experiment 2 web app: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
