/**
 * Exact t-SNE via gradient descent (KL on pairwise affinities).
 */

const EPS = 1e-12;
const PERP_TOL = 1e-5;
const MAX_SIGMA_ITERS = 100;

function entropy(p) {
  let h = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > EPS) h -= p[i] * Math.log2(p[i]);
  }
  return h;
}

function findSigma(distRow, n, targetPerp, i) {
  let sigmaLo = 1e-20;
  let sigmaHi = 1e10;
  const Prow = new Float64Array(n);

  for (let iter = 0; iter < MAX_SIGMA_ITERS; iter++) {
    const sigma = (sigmaLo + sigmaHi) / 2;
    const inv2s2 = 1 / (2 * sigma * sigma);
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) {
        Prow[j] = 0;
        continue;
      }
      const p = Math.exp(-Math.max(distRow[j], 0) * inv2s2);
      Prow[j] = p;
      sum += p;
    }
    if (sum <= EPS) {
      sigmaLo = sigma;
      continue;
    }
    for (let j = 0; j < n; j++) Prow[j] /= sum;
    const perp = Math.pow(2, entropy(Prow));
    if (Math.abs(perp - targetPerp) < PERP_TOL) return sigma;
    if (perp > targetPerp) sigmaHi = sigma;
    else sigmaLo = sigma;
  }
  return (sigmaLo + sigmaHi) / 2;
}

export function computeJointP(D, n, perplexity) {
  const P = new Float64Array(n * n);

  for (let i = 0; i < n; i++) {
    const distRow = new Float64Array(n);
    for (let j = 0; j < n; j++) distRow[j] = D[i * n + j];

    const sigma = findSigma(distRow, n, perplexity, i);
    const inv2s2 = 1 / (2 * sigma * sigma);
    let sum = 0;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const val = Math.exp(-Math.max(distRow[j], 0) * inv2s2);
      P[i * n + j] = val;
      sum += val;
    }
    for (let j = 0; j < n; j++) {
      if (j !== i) P[i * n + j] /= sum;
    }
  }

  const Psym = new Float64Array(n * n);
  const norm = 2 * n;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = (P[i * n + j] + P[j * n + i]) / norm;
      Psym[i * n + j] = v;
      Psym[j * n + i] = v;
    }
  }
  const floor = 1e-12 / (n * n);
  let Psum = 0;
  for (let i = 0; i < n * n; i++) {
    Psym[i] = Math.max(Psym[i], floor);
    Psum += Psym[i];
  }
  for (let i = 0; i < n * n; i++) Psym[i] /= Psum;
  return Psym;
}

export function initEmbedding2D(n, rng, scale = 1e-4) {
  const Y = new Float64Array(n * 2);
  for (let i = 0; i < n * 2; i++) Y[i] = (rng() - 0.5) * scale;
  return Y;
}

function singleStep(state, learningRate, momentum, exaggeration) {
  const { P, n, Y, gains, velocity, prevGrad } = state;
  const Q = new Float64Array(n * n);
  let Qsum = 0;

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = Y[i * 2] - Y[j * 2];
      const dy = Y[i * 2 + 1] - Y[j * 2 + 1];
      const q = 1 / (1 + dx * dx + dy * dy);
      Q[i * n + j] = q;
      Q[j * n + i] = q;
      Qsum += 2 * q;
    }
  }

  const grad = new Float64Array(n * 2);
  let loss = 0;

  // Early exaggeration: scale P then use in KL (gradient uses same scaled mass)
  const pScale = exaggeration;

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const pij = P[i * n + j] * pScale;
      const qij = Math.max(Q[i * n + j] / Qsum, EPS);
      const mult = 4 * (pij - qij) * Q[i * n + j];
      grad[i * 2] += mult * (Y[i * 2] - Y[j * 2]);
      grad[i * 2 + 1] += mult * (Y[i * 2 + 1] - Y[j * 2 + 1]);
      if (i < j) {
        const p = Math.max(pij, EPS);
        loss += p * Math.log(p / qij);
      }
    }
  }

  const mom = state.iter < 250 ? momentum : 0.8;
  const minGain = 0.01;

  for (let d = 0; d < n * 2; d++) {
    const gi = grad[d];
    if (prevGrad) {
      if ((gi > 0) !== (prevGrad[d] > 0)) gains[d] += 0.2;
      else gains[d] *= 0.8;
    }
    gains[d] = Math.max(gains[d], minGain);
    velocity[d] = mom * velocity[d] - learningRate * (gi / gains[d]);
    Y[d] += velocity[d];
    prevGrad[d] = gi;
  }

  return loss;
}

export function tsneStep(state, options = {}) {
  const steps = options.steps ?? 1;
  const learningRate = options.learningRate ?? state.learningRate ?? 200;
  const momentum = options.momentum ?? state.momentum ?? 0.5;
  const earlyExaggeration = options.earlyExaggeration ?? 12;
  const exaggerationEnd = options.exaggerationEnd ?? 100;

  let loss = 0;
  for (let s = 0; s < steps; s++) {
    const exaggeration = state.iter < exaggerationEnd ? earlyExaggeration : 1;
    loss = singleStep(state, learningRate, momentum, exaggeration);
    state.iter += 1;
  }
  state.lastLoss = loss;
  return { Y: state.Y, loss, iter: state.iter };
}

export function scaleEmbedding(Y, n, targetStd = 1e-2) {
  let mx = 0,
    my = 0;
  for (let i = 0; i < n; i++) {
    mx += Y[i * 2];
    my += Y[i * 2 + 1];
  }
  mx /= n;
  my /= n;
  let vx = 0,
    vy = 0;
  for (let i = 0; i < n; i++) {
    const dx = Y[i * 2] - mx;
    const dy = Y[i * 2 + 1] - my;
    vx += dx * dx;
    vy += dy * dy;
  }
  const sx = Math.sqrt(vx / n) || 1;
  const sy = Math.sqrt(vy / n) || 1;
  const s = targetStd / Math.max(sx, sy);
  for (let i = 0; i < n; i++) {
    Y[i * 2] = (Y[i * 2] - mx) * s;
    Y[i * 2 + 1] = (Y[i * 2 + 1] - my) * s;
  }
  return Y;
}

export function createTsneState(P, n, rng, opts = {}) {
  let Y;
  if (opts.initY) {
    Y = opts.initY.slice();
    scaleEmbedding(Y, n, opts.initScale ?? 1e-2);
  } else {
    Y = initEmbedding2D(n, rng);
  }
  return {
    P,
    n,
    Y,
    gains: new Float64Array(n * 2).fill(1),
    velocity: new Float64Array(n * 2).fill(0),
    prevGrad: new Float64Array(n * 2).fill(0),
    iter: 0,
    lastLoss: 0,
    learningRate: opts.learningRate ?? 200,
    momentum: opts.momentum ?? 0.5,
  };
}
