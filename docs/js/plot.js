import { PointKind } from "./data.js";

const CLUSTER_COLORS = {
  [PointKind.CLUSTER_A]: "#3b528b",
  [PointKind.CLUSTER_B]: "#fde725",
};

const INJECTION_COLOR = "#e45756";

export function colorForKind(kind) {
  if (kind === PointKind.OUTLIER || kind === PointKind.POISON) return INJECTION_COLOR;
  return CLUSTER_COLORS[kind] ?? "#666";
}

export function isInjection(kind) {
  return kind === PointKind.OUTLIER || kind === PointKind.POISON;
}

function niceTicks(min, max, targetCount = 5) {
  const range = max - min;
  if (range <= 0) return [min];
  const rough = range / targetCount;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  let step;
  if (norm <= 1) step = mag;
  else if (norm <= 2) step = 2 * mag;
  else if (norm <= 5) step = 5 * mag;
  else step = 10 * mag;
  const ticks = [];
  const start = Math.ceil(min / step) * step;
  for (let v = start; v <= max + step * 0.001; v += step) ticks.push(v);
  return ticks;
}

export class ScatterPlot {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.padding = 44;
    this.Y = null;
    this.kinds = null;
    this.n = 0;
    this.bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    this.plotLeft = 0;
    this.plotTop = 0;
    this.plotSize = 100;
    this.pointRadius = 5;
    this.emptyMessage = "Adjust parameters to preview PCA";
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.floor(rect.width * dpr);
    this.canvas.height = Math.floor(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.updatePlotLayout();
    this.draw();
  }

  setData(Y, kinds, n) {
    this.Y = Y;
    this.kinds = kinds;
    this.n = n ?? (kinds?.length ?? 0);
    if (this.Y && this.n > 0) this.updateBounds();
    this.updatePlotLayout();
    this.draw();
  }

  setEmptyMessage(msg) {
    this.emptyMessage = msg;
    this.draw();
  }

  /** Equal axis ranges: same scale on x and y. */
  updateBounds() {
    if (!this.Y || this.n === 0) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (let i = 0; i < this.n; i++) {
      const x = this.Y[i * 2];
      const y = this.Y[i * 2 + 1];
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const half = Math.max((maxX - minX) / 2, (maxY - minY) / 2, 1e-6) * 1.12;
    this.bounds = {
      minX: cx - half,
      maxX: cx + half,
      minY: cy - half,
      maxY: cy + half,
    };
  }

  updatePlotLayout() {
    const innerW = this.width - 2 * this.padding;
    const innerH = this.height - 2 * this.padding;
    this.plotSize = Math.max(Math.min(innerW, innerH), 1);
    this.plotLeft = this.padding + (innerW - this.plotSize) / 2;
    this.plotTop = this.padding + (innerH - this.plotSize) / 2;
  }

  toScreen(x, y) {
    const { minX, maxX, minY, maxY } = this.bounds;
    const span = maxX - minX || 1;
    const sx = this.plotLeft + ((x - minX) / span) * this.plotSize;
    const sy = this.plotTop + this.plotSize - ((y - minY) / span) * this.plotSize;
    return [sx, sy];
  }

  drawGrid(ctx) {
    const { minX, maxX, minY, maxY } = this.bounds;
    const xTicks = niceTicks(minX, maxX);
    const yTicks = niceTicks(minY, maxY);

    ctx.save();
    ctx.strokeStyle = "#e8e8ec";
    ctx.lineWidth = 1;

    for (const x of xTicks) {
      const [sx] = this.toScreen(x, minY);
      ctx.beginPath();
      ctx.moveTo(sx, this.plotTop);
      ctx.lineTo(sx, this.plotTop + this.plotSize);
      ctx.stroke();
    }

    for (const y of yTicks) {
      const [, sy] = this.toScreen(minX, y);
      ctx.beginPath();
      ctx.moveTo(this.plotLeft, sy);
      ctx.lineTo(this.plotLeft + this.plotSize, sy);
      ctx.stroke();
    }

    ctx.strokeStyle = "#c8c8d0";
    ctx.lineWidth = 1.25;
    ctx.strokeRect(this.plotLeft, this.plotTop, this.plotSize, this.plotSize);
    ctx.restore();
  }

  draw() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, w, h);

    this.updatePlotLayout();

    if (!this.Y || this.n === 0) {
      ctx.fillStyle = "#888";
      ctx.font = "14px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this.emptyMessage, w / 2, h / 2);
      return;
    }

    this.drawGrid(ctx);

    for (let i = 0; i < this.n; i++) {
      const kind = this.kinds[i];
      const [sx, sy] = this.toScreen(this.Y[i * 2], this.Y[i * 2 + 1]);
      const injection = isInjection(kind);
      const r = injection ? 8 : this.pointRadius;
      ctx.globalAlpha = 0.88;
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = colorForKind(kind);
      ctx.fill();
      if (injection) {
        ctx.strokeStyle = "#8b1a1a";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
}
