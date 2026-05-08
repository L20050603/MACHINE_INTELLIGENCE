const els = {
  start: document.querySelector("#start"),
  goal: document.querySelector("#goal"),
  heuristic: document.querySelector("#heuristic"),
  weight: document.querySelector("#weight"),
  maxExpanded: document.querySelector("#maxExpanded"),
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
};

let currentPath = [[[2, 8, 3], [1, 6, 4], [7, 0, 5]]];
let currentMoves = [];
let currentStep = 0;

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

function renderBoard(rows) {
  els.board.innerHTML = "";
  rows.flat().forEach((value) => {
    const tile = document.createElement("div");
    tile.className = value === 0 ? "tile blank" : "tile";
    tile.textContent = value === 0 ? "_" : value;
    els.board.appendChild(tile);
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

function updateStats(result) {
  els.found.textContent = result.found ? "是" : "否";
  els.depth.textContent = result.depth;
  els.expanded.textContent = result.expanded;
  els.generated.textContent = result.generated;
  els.frontier.textContent = result.max_frontier;
  els.elapsed.textContent = `${result.elapsed.toFixed(6)}s`;
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
        <td>${row.heuristic}</td>
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
els.prevBtn.addEventListener("click", () => {
  currentStep -= 1;
  updateStep();
});
els.nextBtn.addEventListener("click", () => {
  currentStep += 1;
  updateStep();
});

updateStep();
