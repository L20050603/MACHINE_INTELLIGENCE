const canvas = document.querySelector("#digitCanvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const preview = document.querySelector("#previewCanvas");
const previewCtx = preview.getContext("2d");
const brushSize = document.querySelector("#brushSize");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const statusEl = document.querySelector("#modelStatus");
const predictionEl = document.querySelector("#prediction");
const confidenceEl = document.querySelector("#confidence");
const barsEl = document.querySelector("#bars");
const topList = document.querySelector("#topList");
const top3Box = document.querySelector("#top3Box");
const donutModal = document.querySelector("#donutModal");
const modalDonut = document.querySelector("#modalDonut");
const modalClose = document.querySelector(".modal-close");
const modelSelect = document.querySelector("#modelSelect");
const modelNote = document.querySelector("#modelNote");
const ganStatus = document.querySelector("#ganStatus");
const ganDigit = document.querySelector("#ganDigit");
const ganGenerateBtn = document.querySelector("#ganGenerateBtn");
const ganImage = document.querySelector("#ganImage");
const ganMessage = document.querySelector("#ganMessage");

let drawing = false;
let last = null;
let generatorReady = false;
let models = [];

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  predictionEl.textContent = "-";
  confidenceEl.textContent = "等待输入";
  showTop3List(null);
  cachedProbabilities = null;
  drawBars(Array(10).fill(0));
  drawPreview(Array(784).fill(0));
  donutModal.classList.remove("open");
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function drawLine(from, to) {
  ctx.strokeStyle = "#05070a";
  ctx.lineWidth = Number(brushSize.value);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
}

function startDraw(event) {
  event.preventDefault();
  drawing = true;
  last = canvasPoint(event);
  drawLine(last, last);
}

function moveDraw(event) {
  if (!drawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  drawLine(last, point);
  last = point;
}

function endDraw() {
  drawing = false;
  last = null;
}

function buildDigitMask() {
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const mask = new Float32Array(canvas.width * canvas.height);
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const offset = (y * canvas.width + x) * 4;
      const gray = (data.data[offset] + data.data[offset + 1] + data.data[offset + 2]) / 765;
      const ink = Math.max(0, Math.min(1, 1 - gray));
      mask[y * canvas.width + x] = ink;
      if (ink > 0.08) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return { mask, minX, minY, maxX, maxY };
}

function preprocess() {
  const digit = buildDigitMask();
  if (!digit) return Array(784).fill(0);

  const cropW = digit.maxX - digit.minX + 1;
  const cropH = digit.maxY - digit.minY + 1;
  const source = document.createElement("canvas");
  source.width = cropW;
  source.height = cropH;
  const sourceCtx = source.getContext("2d", { willReadFrequently: true });
  const sourceImage = sourceCtx.createImageData(cropW, cropH);

  for (let y = 0; y < cropH; y += 1) {
    for (let x = 0; x < cropW; x += 1) {
      const ink = digit.mask[(digit.minY + y) * canvas.width + digit.minX + x];
      const value = Math.round(ink * 255);
      const offset = (y * cropW + x) * 4;
      sourceImage.data[offset] = value;
      sourceImage.data[offset + 1] = value;
      sourceImage.data[offset + 2] = value;
      sourceImage.data[offset + 3] = 255;
    }
  }
  sourceCtx.putImageData(sourceImage, 0, 0);

  const target = document.createElement("canvas");
  target.width = 28;
  target.height = 28;
  const targetCtx = target.getContext("2d", { willReadFrequently: true });
  targetCtx.fillStyle = "black";
  targetCtx.fillRect(0, 0, 28, 28);
  targetCtx.imageSmoothingEnabled = true;
  targetCtx.imageSmoothingQuality = "high";

  const scale = 20 / Math.max(cropW, cropH);
  const drawW = Math.max(1, Math.round(cropW * scale));
  const drawH = Math.max(1, Math.round(cropH * scale));
  const dx = Math.floor((28 - drawW) / 2);
  const dy = Math.floor((28 - drawH) / 2);
  targetCtx.drawImage(source, dx, dy, drawW, drawH);

  const small = targetCtx.getImageData(0, 0, 28, 28);
  const pixels = [];
  for (let i = 0; i < small.data.length; i += 4) {
    pixels.push((small.data[i] + small.data[i + 1] + small.data[i + 2]) / 765);
  }
  return pixels;
}

function drawPreview(pixels) {
  const image = previewCtx.createImageData(28, 28);
  pixels.forEach((v, idx) => {
    const color = Math.round((1 - v) * 255);
    image.data[idx * 4] = color;
    image.data[idx * 4 + 1] = color;
    image.data[idx * 4 + 2] = color;
    image.data[idx * 4 + 3] = 255;
  });

  const temp = document.createElement("canvas");
  temp.width = 28;
  temp.height = 28;
  temp.getContext("2d").putImageData(image, 0, 0);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.clearRect(0, 0, preview.width, preview.height);
  previewCtx.drawImage(temp, 0, 0, preview.width, preview.height);
}

function drawBars(probabilities) {
  barsEl.innerHTML = "";
  probabilities.forEach((probability, digit) => {
    const row = document.createElement("div");
    row.className = "bar";
    row.innerHTML = `
      <strong>${digit}</strong>
      <span class="track"><span class="fill" style="width:${Math.max(0, Math.min(100, probability * 100))}%"></span></span>
      <span>${(probability * 100).toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  });
}

let cachedTop3 = null;
let cachedProbabilities = null;

function showTop3List(top3) {
  cachedTop3 = top3;
  topList.innerHTML = "";

  if (!top3 || top3.length === 0) {
    topList.innerHTML = '<li style="color:var(--muted)">—</li>';
    top3Box.classList.remove("has-results");
    return;
  }

  top3Box.classList.add("has-results");
  const colors = ["#315f9e", "#d4552d", "#0b7a75"];

  top3.forEach((item, i) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <span class="rank-dot" style="background:${colors[i]}"></span>
      <span class="rank-label">${item.label}</span>
      <span class="rank-pct">${(item.probability * 100).toFixed(1)}%</span>
    `;
    topList.appendChild(li);
  });
}

function buildDonutArc(cx, cy, outerR, innerR, cosS, sinS, cosE, sinE, large) {
  const x1 = cx + outerR * cosS, y1 = cy + outerR * sinS;
  const x2 = cx + outerR * cosE, y2 = cy + outerR * sinE;
  const x3 = cx + innerR * cosE, y3 = cy + innerR * sinE;
  const x4 = cx + innerR * cosS, y4 = cy + innerR * sinS;
  return `M${x1.toFixed(1)} ${y1.toFixed(1)} A${outerR} ${outerR} 0 ${large} 1 ${x2.toFixed(1)} ${y2.toFixed(1)} L${x3.toFixed(1)} ${y3.toFixed(1)} A${innerR} ${innerR} 0 ${large} 0 ${x4.toFixed(1)} ${y4.toFixed(1)} Z`;
}

function buildFullDonutPath(cx, cy, outerR, innerR, cosS, sinS, cosM, sinM, cosE, sinE) {
  return buildDonutArc(cx, cy, outerR, innerR, cosS, sinS, cosM, sinM, 1)
    + " " + buildDonutArc(cx, cy, outerR, innerR, cosM, sinM, cosE, sinE, 1);
}

function drawModalDonut(probabilities) {
  modalDonut.innerHTML = "";

  if (!probabilities || probabilities.length === 0) return;

  // Digits with prob >= 1% shown individually, rest grouped as "其他"
  const raw = probabilities
    .map((prob, digit) => ({ label: String(digit), prob }))
    .filter((item) => item.prob > 0.0001)
    .sort((a, b) => b.prob - a.prob);

  const segments = [];
  let otherProb = 0;

  raw.forEach((item) => {
    if (item.prob >= 0.01) {
      segments.push(item);
    } else {
      otherProb += item.prob;
    }
  });

  if (otherProb > 0.0001) {
    segments.push({ label: "其他", prob: otherProb });
  }

  if (segments.length === 0) return;

  const total = segments.reduce((s, item) => s + item.prob, 0);
  const colors = [
    "#315f9e", "#d4552d", "#0b7a75", "#7b4ea3", "#c47d2d",
    "#4a8c5c", "#b8456e", "#5c7a8c", "#9e6b3e", "#6c8cbf",
  ];

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 540 540");

  const cx = 270, cy = 270, outerR = 108, innerR = 54;
  let startAngle = -Math.PI / 2;

  // Five-tier label radii — wider spread to prevent overlap
  const labelTiers = [
    outerR + 42, outerR + 62, outerR + 82, outerR + 102, outerR + 122,
  ];

  segments.forEach((item, i) => {
    const fraction = item.prob / total;
    const sweep = fraction * Math.PI * 2;
    const endAngle = startAngle + sweep;
    const midAngle = startAngle + sweep / 2;

    const cosS = Math.cos(startAngle), sinS = Math.sin(startAngle);
    const cosE = Math.cos(endAngle), sinE = Math.sin(endAngle);
    const cosM = Math.cos(midAngle), sinM = Math.sin(midAngle); // midpoint for full-circle split

    // Full-circle arcs (>= ~358°) must be split into two 180° halves
    // because SVG A commands cannot draw a complete circle (start == end)
    const fullCircle = sweep > Math.PI * 1.999;
    const d = fullCircle
      ? buildFullDonutPath(cx, cy, outerR, innerR, cosS, sinS, cosM, sinM, cosE, sinE)
      : buildDonutArc(cx, cy, outerR, innerR, cosS, sinS, cosE, sinE, sweep > Math.PI ? 1 : 0);

    const path = document.createElementNS(NS, "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", colors[i % colors.length]);
    path.setAttribute("stroke", "#fff");
    path.setAttribute("stroke-width", "2");
    svg.appendChild(path);

    // All labels outside — 5-tier stagger to prevent overlap
    const labelR = labelTiers[i % 5];
    const lx = cx + labelR * Math.cos(midAngle);
    const ly = cy + labelR * Math.sin(midAngle);
    const edgeX = cx + (outerR + 2) * Math.cos(midAngle);
    const edgeY = cy + (outerR + 2) * Math.sin(midAngle);

    // Connector line
    const line = document.createElementNS(NS, "polyline");
    line.setAttribute("points", `${edgeX.toFixed(1)},${edgeY.toFixed(1)} ${lx.toFixed(1)},${ly.toFixed(1)}`);
    line.setAttribute("stroke", "#b0b8c4");
    line.setAttribute("stroke-width", "0.8");
    line.setAttribute("fill", "none");
    svg.appendChild(line);

    const pct = (item.prob * 100).toFixed(1);
    const anchor = lx > cx + 4 ? "start" : lx < cx - 4 ? "end" : "middle";
    const ox = anchor === "start" ? 4 : anchor === "end" ? -4 : 0;

    const t2 = document.createElementNS(NS, "text");
    t2.setAttribute("x", (lx + ox).toFixed(1));
    t2.setAttribute("y", (ly - 16).toFixed(1));
    t2.setAttribute("text-anchor", anchor);
    t2.setAttribute("fill", "#c0392b");
    t2.setAttribute("font-size", "12");
    t2.setAttribute("font-weight", "bold");
    t2.setAttribute("font-family", "Microsoft YaHei, Segoe UI, sans-serif");
    t2.textContent = pct + "%";
    svg.appendChild(t2);

    const t1 = document.createElementNS(NS, "text");
    t1.setAttribute("x", (lx + ox).toFixed(1));
    t1.setAttribute("y", (ly + 4).toFixed(1));
    t1.setAttribute("text-anchor", anchor);
    t1.setAttribute("fill", "#17202a");
    t1.setAttribute("font-size", "15");
    t1.setAttribute("font-weight", "bold");
    t1.setAttribute("font-family", "Microsoft YaHei, Segoe UI, sans-serif");
    t1.textContent = item.label;
    svg.appendChild(t1);

    startAngle = endAngle;
  });

  // Center label
  const center = document.createElementNS(NS, "text");
  center.setAttribute("x", cx.toString());
  center.setAttribute("y", cy.toString());
  center.setAttribute("text-anchor", "middle");
  center.setAttribute("dominant-baseline", "central");
  center.setAttribute("fill", "#17202a");
  center.setAttribute("font-size", "15");
  center.setAttribute("font-weight", "bold");
  center.setAttribute("font-family", "Microsoft YaHei, Segoe UI, sans-serif");
  center.textContent = segments.length <= 5 ? "概率分布" : "10 类";
  svg.appendChild(center);

  modalDonut.appendChild(svg);
}

function openModal() {
  if (!cachedProbabilities || cachedProbabilities.length === 0) return;
  drawModalDonut(cachedProbabilities);
  donutModal.classList.add("open");
  donutModal.setAttribute("aria-hidden", "false");
}

function closeModal() {
  donutModal.classList.remove("open");
  donutModal.setAttribute("aria-hidden", "true");
}

function selectedModel() {
  return modelSelect.value;
}

function updateModelNote() {
  const model = models.find((item) => item.id === selectedModel());
  modelNote.textContent = model ? model.note : "未选择模型";
}

async function predict() {
  const pixels = preprocess();
  drawPreview(pixels);
  predictBtn.disabled = true;
  confidenceEl.textContent = "识别中";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels, modelId: selectedModel() }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "识别失败");

    const model = models.find((item) => item.id === result.modelId);
    predictionEl.textContent = result.prediction;
    confidenceEl.textContent = `${model ? model.name : result.modelId}，置信度 ${(result.confidence * 100).toFixed(2)}%`;
    drawBars(result.probabilities);
    showTop3List(result.top3);
    cachedProbabilities = result.probabilities;
  } catch (error) {
    predictionEl.textContent = "!";
    confidenceEl.textContent = error.message;
    showTop3List(null);
    cachedProbabilities = null;
  } finally {
    predictBtn.disabled = false;
  }
}

async function generateDigit() {
  if (!generatorReady) return;
  ganGenerateBtn.disabled = true;
  ganMessage.textContent = "生成中";
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ digit: Number(ganDigit.value), count: 6 }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "生成失败");
    ganImage.src = result.image;
    ganMessage.textContent = `数字 ${result.digit} 的合成样本`;
  } catch (error) {
    ganMessage.textContent = error.message;
    ganImage.removeAttribute("src");
  } finally {
    ganGenerateBtn.disabled = false;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const result = await response.json();
    models = result.models || [];

    modelSelect.innerHTML = models
      .map((model) => `<option value="${model.id}" ${model.ready ? "" : "disabled"}>${model.name}${model.ready ? "" : "（未训练）"}</option>`)
      .join("");
    modelSelect.value = result.defaultModel;
    if (!models.some((model) => model.ready)) {
      predictBtn.disabled = true;
      modelNote.textContent = "未找到可用权重";
    } else {
      updateModelNote();
    }

    statusEl.textContent = result.modelReady ? "模型已加载" : "未训练";
    statusEl.className = `status ${result.modelReady ? "ready" : "warn"}`;

    generatorReady = Boolean(result.generatorReady);
    ganStatus.textContent = generatorReady ? "已加载" : "未训练";
    ganStatus.className = `pill ${generatorReady ? "ready" : "warn"}`;
    ganGenerateBtn.disabled = !generatorReady;
    ganMessage.textContent = generatorReady ? "选择数字生成" : "未找到 ACGAN 权重";
  } catch {
    models = [];
    modelSelect.innerHTML = "";
    predictionEl.textContent = "-";
    confidenceEl.textContent = "等待输入";
    statusEl.textContent = "离线";
    statusEl.className = "status warn";
    modelNote.textContent = "服务未连接";
    ganStatus.textContent = "离线";
    ganStatus.className = "pill warn";
    ganGenerateBtn.disabled = true;
    generatorReady = false;
    predictBtn.disabled = true;
  }
}

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);
clearBtn.addEventListener("click", resetCanvas);
predictBtn.addEventListener("click", predict);
modelSelect.addEventListener("change", updateModelNote);
ganGenerateBtn.addEventListener("click", generateDigit);

top3Box.addEventListener("click", openModal);
modalClose.addEventListener("click", closeModal);
donutModal.addEventListener("click", (e) => {
  if (e.target === donutModal) closeModal();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && donutModal.classList.contains("open")) closeModal();
});

resetCanvas();
checkHealth();

