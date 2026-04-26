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
  topList.innerHTML = "";
  drawBars(Array(10).fill(0));
  drawPreview(Array(784).fill(0));
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
    topList.innerHTML = result.top3
      .map((item) => `<li>${item.label}: ${(item.probability * 100).toFixed(2)}%</li>`)
      .join("");
  } catch (error) {
    predictionEl.textContent = "!";
    confidenceEl.textContent = error.message;
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
    statusEl.textContent = "离线";
    statusEl.className = "status warn";
    modelNote.textContent = "服务未连接";
    ganStatus.textContent = "离线";
    ganStatus.className = "pill warn";
    ganGenerateBtn.disabled = true;
  }
}

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);
clearBtn.addEventListener("click", resetCanvas);
predictBtn.addEventListener("click", predict);
modelSelect.addEventListener("change", updateModelNote);
ganGenerateBtn.addEventListener("click", generateDigit);

resetCanvas();
checkHealth();

