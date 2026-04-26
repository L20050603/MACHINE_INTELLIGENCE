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

let drawing = false;
let last = null;

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
  const client = event.touches ? event.touches[0] : event;
  return {
    x: ((client.clientX - rect.left) / rect.width) * canvas.width,
    y: ((client.clientY - rect.top) / rect.height) * canvas.height,
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

function preprocess() {
  const data = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Float32Array(canvas.width * canvas.height);
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = 0;
  let maxY = 0;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const i = (y * canvas.width + x) * 4;
      const value = 1 - (data.data[i] + data.data[i + 1] + data.data[i + 2]) / 765;
      gray[y * canvas.width + x] = value;
      if (value > 0.08) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX <= minX || maxY <= minY) return Array(784).fill(0);

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;
  const scale = 20 / Math.max(cropW, cropH);
  const drawW = Math.max(1, Math.round(cropW * scale));
  const drawH = Math.max(1, Math.round(cropH * scale));

  const temp = document.createElement("canvas");
  temp.width = 28;
  temp.height = 28;
  const tctx = temp.getContext("2d", { willReadFrequently: true });
  tctx.fillStyle = "black";
  tctx.fillRect(0, 0, 28, 28);
  tctx.drawImage(canvas, minX, minY, cropW, cropH, Math.floor((28 - drawW) / 2), Math.floor((28 - drawH) / 2), drawW, drawH);

  const small = tctx.getImageData(0, 0, 28, 28);
  const pixels = [];
  for (let i = 0; i < small.data.length; i += 4) {
    pixels.push(1 - (small.data[i] + small.data[i + 1] + small.data[i + 2]) / 765);
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

async function predict() {
  const pixels = preprocess();
  drawPreview(pixels);
  predictBtn.disabled = true;
  confidenceEl.textContent = "识别中";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pixels }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "识别失败");

    predictionEl.textContent = result.prediction;
    confidenceEl.textContent = `置信度 ${(result.confidence * 100).toFixed(2)}%`;
    drawBars(result.probabilities);
    topList.innerHTML = result.top3
      .map((item) => `<li>${item.label}：${(item.probability * 100).toFixed(2)}%</li>`)
      .join("");
    if (!result.modelReady) {
      confidenceEl.textContent += "，当前未加载训练权重";
    }
  } catch (error) {
    predictionEl.textContent = "!";
    confidenceEl.textContent = error.message;
  } finally {
    predictBtn.disabled = false;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const result = await response.json();
    statusEl.textContent = result.modelReady ? "模型已加载" : "未训练";
    statusEl.className = `status ${result.modelReady ? "ready" : "warn"}`;
  } catch {
    statusEl.textContent = "离线";
    statusEl.className = "status warn";
  }
}

canvas.addEventListener("pointerdown", startDraw);
canvas.addEventListener("pointermove", moveDraw);
window.addEventListener("pointerup", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
window.addEventListener("touchend", endDraw);
clearBtn.addEventListener("click", resetCanvas);
predictBtn.addEventListener("click", predict);

resetCanvas();
checkHealth();

