const DEFAULT_START = "2 8 3 1 6 4 7 0 5";
const DEFAULT_GOAL = "1 2 3 8 0 4 7 6 5";
const CANONICAL_GOAL = [1, 2, 3, 4, 5, 6, 7, 8, 0];

const els = {
  start: document.querySelector("#start"),
  goal: document.querySelector("#goal"),
  heuristic: document.querySelector("#heuristic"),
  weight: document.querySelector("#weight"),
  maxExpanded: document.querySelector("#maxExpanded"),
  randomStartBtn: document.querySelector("#randomStartBtn"),
  randomPairBtn: document.querySelector("#randomPairBtn"),
  swapBtn: document.querySelector("#swapBtn"),
  resetBtn: document.querySelector("#resetBtn"),
  solveBtn: document.querySelector("#solveBtn"),
  compareBtn: document.querySelector("#compareBtn"),
  status: document.querySelector("#status"),
  board: document.querySelector("#board"),
  stepLabel: document.querySelector("#stepLabel"),
  moveLabel: document.querySelector("#moveLabel"),
  prevBtn: document.querySelector("#prevBtn"),
  nextBtn: document.querySelector("#nextBtn"),
  found: document.querySelector("#found"),
  depth: document.querySelector("#depth"),
  expanded: document.querySelector("#expanded"),
  generated: document.querySelector("#generated"),
  frontier: document.querySelector("#frontier"),
  elapsed: document.querySelector("#elapsed"),
  compareBody: document.querySelector("#compareBody"),
  modeBadge: document.querySelector("#modeBadge"),
};

let currentPath = [[[2, 8, 3], [1, 6, 4], [7, 0, 5]]];
let currentMoves = [];
let currentStep = 0;
let boardCells = [];

function payload() {
  return {
    start: els.start.value,
    goal: els.goal.value,
    heuristic: els.heuristic.value,
    weight: Number(els.weight.value),
    max_expanded: Number(els.maxExpanded.value),
  };
}

async function postJson(url, data) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "请求失败");
  }
  return result;
}

function parseStateText(raw) {
  const text = raw.trim();
  if (/^\d{9}$/.test(text)) {
    return [...text].map(Number);
  }
  return text
    .replace(/[,;]/g, " ")
    .split(/\s+/)
    .filter(Boolean)
    .map(Number);
}

function formatState(values) {
  return values.join(" ");
}

function rowsFromFlat(values) {
  return [values.slice(0, 3), values.slice(3, 6), values.slice(6, 9)];
}

function readGoalOrDefault() {
  const values = parseStateText(els.goal.value);
  return values.length === 9 ? values : CANONICAL_GOAL;
}

function neighborStates(state) {
  const blank = state.indexOf(0);
  const row = Math.floor(blank / 3);
  const col = blank % 3;
  const targets = [];
  if (row > 0) targets.push(blank - 3);
  if (row < 2) targets.push(blank + 3);
  if (col > 0) targets.push(blank - 1);
  if (col < 2) targets.push(blank + 1);
  return targets.map((target) => {
    const next = [...state];
    [next[blank], next[target]] = [next[target], next[blank]];
    return next;
  });
}

function scramble(base, steps) {
  let state = [...base];
  let previous = null;
  for (let i = 0; i < steps; i += 1) {
    const choices = neighborStates(state).filter((item) => item.join(",") !== previous);
    previous = state.join(",");
    state = choices[Math.floor(Math.random() * choices.length)];
  }
  return state;
}

function randomSteps(min = 18, max = 46) {
  return min + Math.floor(Math.random() * (max - min + 1));
}

function setPreviewFromStart() {
  const values = parseStateText(els.start.value);
  if (values.length === 9) {
    currentPath = [rowsFromFlat(values)];
    currentMoves = [];
    currentStep = 0;
    updateStep();
  }
}

function randomizeStart() {
  const goal = readGoalOrDefault();
  const nextStart = scramble(goal, randomSteps());
  els.start.value = formatState(nextStart);
  setPreviewFromStart();
  clearBusy("已生成随机可解初态。");
}

function randomizePair() {
  const nextGoal = scramble(CANONICAL_GOAL, randomSteps(10, 28));
  const nextStart = scramble(nextGoal, randomSteps());
  els.goal.value = formatState(nextGoal);
  els.start.value = formatState(nextStart);
  setPreviewFromStart();
  clearBusy("已生成随机任务。");
}

function swapStates() {
  [els.start.value, els.goal.value] = [els.goal.value, els.start.value];
  setPreviewFromStart();
  clearBusy("初始状态和目标状态已交换。");
}

function resetStates() {
  els.start.value = DEFAULT_START;
  els.goal.value = DEFAULT_GOAL;
  els.heuristic.value = "pattern_db";
  els.weight.value = "1.0";
  els.maxExpanded.value = "200000";
  setPreviewFromStart();
  updateStats();
  els.compareBody.innerHTML = '<tr><td colspan="5">点击“对比启发函数”。</td></tr>';
  clearBusy("已恢复默认任务。");
}

function initBoard() {
  els.board.innerHTML = "";
  boardCells = [];
  for (let index = 0; index < 9; index += 1) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.style.setProperty("--index", index);
    els.board.appendChild(tile);
    boardCells.push(tile);
  }
}

function renderBoard(rows) {
  rows.flat().forEach((value, index) => {
    const tile = boardCells[index];
    if (!tile) {
      return;
    }
    tile.classList.toggle("blank", value === 0);
    tile.textContent = value === 0 ? "" : value;
  });
}

function updateStep() {
  const total = Math.max(0, currentPath.length - 1);
  currentStep = Math.min(Math.max(currentStep, 0), total);
  renderBoard(currentPath[currentStep]);
  els.stepLabel.textContent = `Step ${currentStep} / ${total}`;
  els.moveLabel.textContent = currentMoves.length
    ? `移动序列：${currentMoves.join(" → ")}`
    : "移动序列会显示在这里。";
}

function updateStats(result = null) {
  els.found.textContent = result ? (result.found ? "是" : "否") : "-";
  els.depth.textContent = result ? result.depth : "-";
  els.expanded.textContent = result ? result.expanded : "-";
  els.generated.textContent = result ? result.generated : "-";
  els.frontier.textContent = result ? result.max_frontier : "-";
  els.elapsed.textContent = result ? `${result.elapsed.toFixed(6)}s` : "-";
  els.modeBadge.textContent = result ? result.heuristic.toUpperCase() : "READY";
}

function setBusy(text) {
  els.status.textContent = text;
  els.solveBtn.disabled = true;
  els.compareBtn.disabled = true;
}

function clearBusy(text) {
  els.status.textContent = text;
  els.solveBtn.disabled = false;
  els.compareBtn.disabled = false;
}

async function solve() {
  try {
    setBusy("正在求解...");
    const result = await postJson("/api/solve", payload());
    updateStats(result);
    currentPath = result.path.length ? result.path : [result.start];
    currentMoves = result.moves;
    currentStep = 0;
    updateStep();
    clearBusy(result.found ? "求解完成。" : result.message || "未找到解。");
  } catch (error) {
    clearBusy(`错误：${error.message}`);
  }
}

async function compare() {
  try {
    setBusy("正在对比启发函数...");
    const result = await postJson("/api/compare", payload());
    els.compareBody.innerHTML = "";
    result.rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="heuristic-name">${row.heuristic}</span></td>
        <td>${row.found ? row.depth : "-"}</td>
        <td>${row.expanded}</td>
        <td>${row.generated}</td>
        <td>${row.elapsed.toFixed(6)}s</td>
      `;
      els.compareBody.appendChild(tr);
    });
    clearBusy("启发函数对比完成。");
  } catch (error) {
    clearBusy(`错误：${error.message}`);
  }
}

els.solveBtn.addEventListener("click", solve);
els.compareBtn.addEventListener("click", compare);
els.randomStartBtn.addEventListener("click", randomizeStart);
els.randomPairBtn.addEventListener("click", randomizePair);
els.swapBtn.addEventListener("click", swapStates);
els.resetBtn.addEventListener("click", resetStates);
els.prevBtn.addEventListener("click", () => {
  currentStep -= 1;
  updateStep();
});
els.nextBtn.addEventListener("click", () => {
  currentStep += 1;
  updateStep();
});
els.start.addEventListener("change", setPreviewFromStart);

initBoard();
updateStep();
