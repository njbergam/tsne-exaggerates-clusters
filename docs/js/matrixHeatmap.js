/** Heatmap for t-SNE joint affinity matrix P (n × n). */

import { PointKind } from "./data.js";

function isInjection(kind) {
  return kind === PointKind.OUTLIER || kind === PointKind.POISON;
}

export function orderIndicesByKind(kinds) {
  const order = [];
  for (let g = 0; g <= 3; g++) {
    for (let i = 0; i < kinds.length; i++) {
      if (kinds[i] === g) order.push(i);
    }
  }
  return order;
}

export function permuteSquareMatrix(P, n, order) {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    const oi = order[i];
    for (let j = 0; j < n; j++) {
      out[i * n + j] = P[oi * n + order[j]];
    }
  }
  return out;
}

/** Grayscale value with red tint when row/col is outlier or poison. */
function colorCell(g, rowInj, colInj) {
  if (!rowInj && !colInj) {
    return [g, g, g];
  }
  const t = rowInj && colInj ? 0.5 : 0.32;
  const r = Math.min(255, Math.floor(g * (1 - t) + 235 * t));
  const gb = Math.floor(g * (1 - t * 0.88));
  return [r, gb, gb];
}

export class MatrixHeatmap {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.padding = 8;
    this.P = null;
    this.n = 0;
    this.sortedKinds = null;
    this._offscreen = document.createElement("canvas");
    this._offctx = this._offscreen.getContext("2d");
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.floor(rect.width * dpr);
    this.canvas.height = Math.floor(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = rect.width;
    this.height = rect.height;
    this.draw();
  }

  /** sortedKinds[i] = point kind for row/column i in the displayed matrix. */
  setMatrix(P, n, sortedKinds) {
    this.P = P;
    this.n = n;
    this.sortedKinds = sortedKinds;
    this.draw();
  }

  clear(message = "—") {
    this.P = null;
    this.n = 0;
    this.sortedKinds = null;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, this.width, this.height);
    ctx.fillStyle = "#888";
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(message, this.width / 2, this.height / 2);
  }

  draw() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, w, h);

    if (!this.P || this.n === 0) {
      this.clear();
      return;
    }

    const n = this.n;
    const P = this.P;
    const kinds = this.sortedKinds;
    let max = 0;
    for (let i = 0; i < n * n; i++) {
      if (P[i] > max) max = P[i];
    }
    if (max <= 0) max = 1;

    this._offscreen.width = n;
    this._offscreen.height = n;
    const img = this._offctx.createImageData(n, n);
    const data = img.data;

    for (let i = 0; i < n; i++) {
      const rowInj = kinds && isInjection(kinds[i]);
      for (let j = 0; j < n; j++) {
        const colInj = kinds && isInjection(kinds[j]);
        const v = P[i * n + j] / max;
        const g = Math.floor(255 * Math.pow(v, 0.5));
        const [r, gr, b] = colorCell(g, rowInj, colInj);
        const k = (i * n + j) * 4;
        data[k] = r;
        data[k + 1] = gr;
        data[k + 2] = b;
        data[k + 3] = 255;
      }
    }
    this._offctx.putImageData(img, 0, 0);

    const side = Math.min(w, h) - 2 * this.padding;
    const left = this.padding + (w - 2 * this.padding - side) / 2;
    const top = this.padding + (h - 2 * this.padding - side) / 2;

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this._offscreen, 0, 0, n, n, left, top, side, side);

    ctx.strokeStyle = "#c8c8d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, side, side);
  }
}
