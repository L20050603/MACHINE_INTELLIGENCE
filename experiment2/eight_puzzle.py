from __future__ import annotations

import argparse
import heapq
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

State = Tuple[int, ...]
Move = str
PatternKey = Tuple[int, ...]

BOARD_SIZE = 3
TILE_COUNT = BOARD_SIZE * BOARD_SIZE
DEFAULT_GOAL: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)
DEFAULT_PATTERNS: Tuple[Tuple[int, ...], ...] = ((1, 2, 3, 4), (5, 6, 7, 8))
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class SearchResult:
    found: bool
    start: State
    goal: State
    heuristic: str
    weight: float
    moves: List[Move]
    path: List[State]
    expanded: int
    generated: int
    max_frontier: int
    elapsed: float
    message: str = ""

    @property
    def depth(self) -> int:
        return len(self.moves)


def parse_state(raw: str) -> State:
    """Parse states like '1 2 3 4 5 6 7 8 0', '1,2,...,0' or '123456780'."""
    text = raw.strip()
    if not text:
        raise ValueError("state cannot be empty")

    if len(text) == TILE_COUNT and text.isdigit():
        values = [int(ch) for ch in text]
    else:
        normalized = text.replace(",", " ").replace(";", " ")
        values = [int(part) for part in normalized.split()]

    state = tuple(values)
    validate_state(state)
    return state


def validate_state(state: Sequence[int]) -> None:
    if len(state) != TILE_COUNT:
        raise ValueError(f"state must contain {TILE_COUNT} numbers")
    if sorted(state) != list(range(TILE_COUNT)):
        raise ValueError("state must contain each number from 0 to 8 exactly once")


def board_lines(state: State) -> List[str]:
    lines = []
    for row in range(BOARD_SIZE):
        values = state[row * BOARD_SIZE : (row + 1) * BOARD_SIZE]
        lines.append(" ".join("_" if value == 0 else str(value) for value in values))
    return lines


def format_board(state: State) -> str:
    return "\n".join(board_lines(state))


def inversion_parity_in_goal_order(state: State, goal: State) -> int:
    rank = {tile: index for index, tile in enumerate(goal) if tile != 0}
    seq = [rank[tile] for tile in state if tile != 0]
    inversions = 0
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            if seq[i] > seq[j]:
                inversions += 1
    return inversions % 2


def is_solvable(start: State, goal: State) -> bool:
    validate_state(start)
    validate_state(goal)
    return inversion_parity_in_goal_order(start, goal) == inversion_parity_in_goal_order(goal, goal)


def goal_positions(goal: State) -> Dict[int, Tuple[int, int]]:
    return {tile: divmod(index, BOARD_SIZE) for index, tile in enumerate(goal)}


def misplaced_tiles(state: State, goal: State) -> int:
    return sum(1 for tile, target in zip(state, goal) if tile != 0 and tile != target)


def manhattan_distance(state: State, goal: State) -> int:
    positions = goal_positions(goal)
    total = 0
    for index, tile in enumerate(state):
        if tile == 0:
            continue
        row, col = divmod(index, BOARD_SIZE)
        goal_row, goal_col = positions[tile]
        total += abs(row - goal_row) + abs(col - goal_col)
    return total


def linear_conflict(state: State, goal: State) -> int:
    positions = goal_positions(goal)
    conflicts = 0

    for row in range(BOARD_SIZE):
        row_tiles = state[row * BOARD_SIZE : (row + 1) * BOARD_SIZE]
        for i in range(BOARD_SIZE):
            a = row_tiles[i]
            if a == 0 or positions[a][0] != row:
                continue
            for j in range(i + 1, BOARD_SIZE):
                b = row_tiles[j]
                if b == 0 or positions[b][0] != row:
                    continue
                if positions[a][1] > positions[b][1]:
                    conflicts += 1

    for col in range(BOARD_SIZE):
        col_tiles = [state[row * BOARD_SIZE + col] for row in range(BOARD_SIZE)]
        for i in range(BOARD_SIZE):
            a = col_tiles[i]
            if a == 0 or positions[a][1] != col:
                continue
            for j in range(i + 1, BOARD_SIZE):
                b = col_tiles[j]
                if b == 0 or positions[b][1] != col:
                    continue
                if positions[a][0] > positions[b][0]:
                    conflicts += 1

    return manhattan_distance(state, goal) + 2 * conflicts


def adjacent_positions(position: int) -> Iterable[int]:
    row, col = divmod(position, BOARD_SIZE)
    if row > 0:
        yield position - BOARD_SIZE
    if row < BOARD_SIZE - 1:
        yield position + BOARD_SIZE
    if col > 0:
        yield position - 1
    if col < BOARD_SIZE - 1:
        yield position + 1


def pattern_key(state: State, pattern: Tuple[int, ...]) -> PatternKey:
    positions = {tile: index for index, tile in enumerate(state)}
    return tuple(positions[tile] for tile in pattern) + (positions[0],)


def pattern_db_path(goal: State, pattern: Tuple[int, ...]) -> Path:
    goal_text = "".join(str(value) for value in goal)
    pattern_text = "".join(str(value) for value in pattern)
    return ARTIFACT_DIR / f"pdb_goal_{goal_text}_pattern_{pattern_text}.json"


def build_pattern_database(goal: State, pattern: Tuple[int, ...]) -> Dict[PatternKey, int]:
    """Build one cost-partitioned pattern database with 0-1 Dijkstra.

    A move costs 1 only when the moved tile belongs to the pattern; moving a
    non-pattern tile costs 0. This allows disjoint pattern databases to be added.
    """
    start_key = pattern_key(goal, pattern)
    distances: Dict[PatternKey, int] = {start_key: 0}
    queue: List[Tuple[int, int, PatternKey]] = [(0, 0, start_key)]
    counter = 0

    while queue:
        cost, _, key = heapq.heappop(queue)
        if cost != distances[key]:
            continue

        tile_positions = key[:-1]
        blank = key[-1]
        occupied = {position: index for index, position in enumerate(tile_positions)}

        for target in adjacent_positions(blank):
            next_positions = list(tile_positions)
            step_cost = 0
            if target in occupied:
                pattern_index = occupied[target]
                next_positions[pattern_index] = blank
                step_cost = 1

            next_key = tuple(next_positions) + (target,)
            next_cost = cost + step_cost
            if next_cost >= distances.get(next_key, 10**9):
                continue

            distances[next_key] = next_cost
            counter += 1
            heapq.heappush(queue, (next_cost, counter, next_key))

    return distances


def load_or_build_pattern_database(goal: State, pattern: Tuple[int, ...]) -> Dict[PatternKey, int]:
    path = pattern_db_path(goal, pattern)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {tuple(int(part) for part in key.split(",")): value for key, value in raw.items()}

    database = build_pattern_database(goal, pattern)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {",".join(str(part) for part in key): value for key, value in database.items()}
    path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
    return database


class PatternDatabaseHeuristic:
    def __init__(self, goal: State, patterns: Tuple[Tuple[int, ...], ...] = DEFAULT_PATTERNS):
        self.goal = goal
        self.patterns = patterns
        self.databases = [load_or_build_pattern_database(goal, pattern) for pattern in patterns]

    def __call__(self, state: State, goal: State) -> int:
        if goal != self.goal:
            self.goal = goal
            self.databases = [load_or_build_pattern_database(goal, pattern) for pattern in self.patterns]
        return sum(database[pattern_key(state, pattern)] for pattern, database in zip(self.patterns, self.databases))


HEURISTICS: Dict[str, Callable[[State, State], int]] = {
    "misplaced": misplaced_tiles,
    "manhattan": manhattan_distance,
    "linear_conflict": linear_conflict,
}

STATIC_HEURISTICS = tuple(HEURISTICS)


def neighbors(state: State) -> Iterable[Tuple[State, Move]]:
    blank = state.index(0)
    row, col = divmod(blank, BOARD_SIZE)
    candidates = [
        ("Up", row > 0, blank - BOARD_SIZE),
        ("Down", row < BOARD_SIZE - 1, blank + BOARD_SIZE),
        ("Left", col > 0, blank - 1),
        ("Right", col < BOARD_SIZE - 1, blank + 1),
    ]
    for move, allowed, target in candidates:
        if not allowed:
            continue
        values = list(state)
        values[blank], values[target] = values[target], values[blank]
        yield tuple(values), move


def reconstruct_path(
    parent: Dict[State, Tuple[Optional[State], Optional[Move]]], goal: State
) -> Tuple[List[State], List[Move]]:
    path: List[State] = []
    moves: List[Move] = []
    current: Optional[State] = goal

    while current is not None:
        path.append(current)
        previous, move = parent[current]
        if move is not None:
            moves.append(move)
        current = previous

    path.reverse()
    moves.reverse()
    return path, moves


def solve_astar(
    start: State,
    goal: State = DEFAULT_GOAL,
    heuristic: str = "linear_conflict",
    weight: float = 1.0,
    max_expanded: int = 200_000,
) -> SearchResult:
    validate_state(start)
    validate_state(goal)
    if heuristic not in HEURISTICS and heuristic != "pattern_db":
        raise ValueError(f"unknown heuristic: {heuristic}")
    if weight <= 0:
        raise ValueError("weight must be positive")

    started_at = time.perf_counter()
    if not is_solvable(start, goal):
        return SearchResult(
            found=False,
            start=start,
            goal=goal,
            heuristic=heuristic,
            weight=weight,
            moves=[],
            path=[],
            expanded=0,
            generated=0,
            max_frontier=0,
            elapsed=time.perf_counter() - started_at,
            message="Initial state and goal state have different inversion parity.",
        )

    h = PatternDatabaseHeuristic(goal) if heuristic == "pattern_db" else HEURISTICS[heuristic]
    frontier: List[Tuple[float, int, int, State]] = []
    counter = 0
    start_h = h(start, goal)
    heapq.heappush(frontier, (weight * start_h, 0, counter, start))

    best_g: Dict[State, int] = {start: 0}
    parent: Dict[State, Tuple[Optional[State], Optional[Move]]] = {start: (None, None)}
    expanded = 0
    generated = 1
    max_frontier = 1

    while frontier:
        _, g, _, state = heapq.heappop(frontier)
        if g != best_g.get(state):
            continue

        if state == goal:
            path, moves = reconstruct_path(parent, goal)
            return SearchResult(
                found=True,
                start=start,
                goal=goal,
                heuristic=heuristic,
                weight=weight,
                moves=moves,
                path=path,
                expanded=expanded,
                generated=generated,
                max_frontier=max_frontier,
                elapsed=time.perf_counter() - started_at,
            )

        expanded += 1
        if expanded > max_expanded:
            return SearchResult(
                found=False,
                start=start,
                goal=goal,
                heuristic=heuristic,
                weight=weight,
                moves=[],
                path=[],
                expanded=expanded,
                generated=generated,
                max_frontier=max_frontier,
                elapsed=time.perf_counter() - started_at,
                message=f"Search stopped after expanding {max_expanded} states.",
            )

        for next_state, move in neighbors(state):
            next_g = g + 1
            if next_g >= best_g.get(next_state, 10**9):
                continue
            best_g[next_state] = next_g
            parent[next_state] = (state, move)
            counter += 1
            next_f = next_g + weight * h(next_state, goal)
            heapq.heappush(frontier, (next_f, next_g, counter, next_state))
            generated += 1

        max_frontier = max(max_frontier, len(frontier))

    return SearchResult(
        found=False,
        start=start,
        goal=goal,
        heuristic=heuristic,
        weight=weight,
        moves=[],
        path=[],
        expanded=expanded,
        generated=generated,
        max_frontier=max_frontier,
        elapsed=time.perf_counter() - started_at,
        message="No solution found.",
    )


def scramble(goal: State = DEFAULT_GOAL, steps: int = 30, seed: Optional[int] = None) -> State:
    rng = random.Random(seed)
    state = goal
    previous: Optional[State] = None
    for _ in range(steps):
        choices = [(next_state, move) for next_state, move in neighbors(state) if next_state != previous]
        previous = state
        state, _ = rng.choice(choices)
    return state


def print_result(result: SearchResult, show_path: bool = True) -> None:
    print(f"Heuristic: {result.heuristic}")
    print(f"Weight: {result.weight:g}")
    print(f"Found: {result.found}")
    if result.message:
        print(f"Message: {result.message}")
    print(f"Depth: {result.depth}")
    print(f"Expanded: {result.expanded}")
    print(f"Generated: {result.generated}")
    print(f"Max frontier: {result.max_frontier}")
    print(f"Elapsed: {result.elapsed:.6f}s")

    if not result.found:
        return

    print("Moves:", " ".join(result.moves) if result.moves else "(already solved)")
    if show_path:
        for index, state in enumerate(result.path):
            print()
            print(f"Step {index}")
            print(format_board(state))


def compare_heuristics(start: State, goal: State, weight: float, max_expanded: int) -> None:
    print("heuristic,found,depth,expanded,generated,max_frontier,elapsed")
    for name in list(STATIC_HEURISTICS) + ["pattern_db"]:
        result = solve_astar(
            start=start,
            goal=goal,
            heuristic=name,
            weight=weight,
            max_expanded=max_expanded,
        )
        print(
            f"{name},{result.found},{result.depth},{result.expanded},"
            f"{result.generated},{result.max_frontier},{result.elapsed:.6f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve the 8-puzzle with A* search.")
    parser.add_argument(
        "--start",
        default="2 8 3 1 6 4 7 0 5",
        help="Initial state. Accepts '2 8 3 1 6 4 7 0 5' or '283164705'.",
    )
    parser.add_argument(
        "--goal",
        default="1 2 3 8 0 4 7 6 5",
        help="Goal state. Accepts the same format as --start.",
    )
    parser.add_argument(
        "--heuristic",
        choices=sorted(list(HEURISTICS) + ["pattern_db"]),
        default="pattern_db",
        help="Heuristic used by A*.",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Use 1.0 for standard A*. Larger values enable Weighted A*.",
    )
    parser.add_argument(
        "--max-expanded",
        type=int,
        default=200_000,
        help="Maximum number of states to expand before stopping.",
    )
    parser.add_argument("--no-path", action="store_true", help="Only print summary statistics.")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all heuristics and print a CSV comparison.",
    )
    parser.add_argument(
        "--random",
        type=int,
        metavar="STEPS",
        help="Generate a solvable start state by scrambling the goal for STEPS moves.",
    )
    parser.add_argument("--seed", type=int, help="Random seed used with --random.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    goal = parse_state(args.goal)
    start = scramble(goal=goal, steps=args.random, seed=args.seed) if args.random else parse_state(args.start)

    print("Start:")
    print(format_board(start))
    print()
    print("Goal:")
    print(format_board(goal))
    print()

    if args.compare:
        compare_heuristics(start=start, goal=goal, weight=args.weight, max_expanded=args.max_expanded)
        return

    result = solve_astar(
        start=start,
        goal=goal,
        heuristic=args.heuristic,
        weight=args.weight,
        max_expanded=args.max_expanded,
    )
    print_result(result, show_path=not args.no_path)


if __name__ == "__main__":
    main()
