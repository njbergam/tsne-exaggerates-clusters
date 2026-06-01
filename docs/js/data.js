/**
 * Synthetic data matching utils.py: TwoGaussians, append_outliers, add_poison.
 */

import { randn } from "./math.js";

export const PointKind = {
  CLUSTER_A: 0,
  CLUSTER_B: 1,
  OUTLIER: 2,
  POISON: 3,
};

/**
 * Two high-dimensional Gaussian clusters (separated along axis 0).
 */
export function twoGaussians({
  nPerCluster = 75,
  distance = 2,
  dim = 50,
  power = -0.75,
  c = 1.0,
  rng,
}) {
  const n = 2 * nPerCluster;
  const scale = c * Math.pow(dim, power);
  const X = [];
  const kinds = [];

  const meanB = new Float64Array(dim);
  meanB[0] = distance;

  for (let k = 0; k < nPerCluster; k++) {
    const row = sampleGaussian(dim, scale, rng, null);
    X.push(row);
    kinds.push(PointKind.CLUSTER_A);
  }
  for (let k = 0; k < nPerCluster; k++) {
    const row = sampleGaussian(dim, scale, rng, meanB);
    X.push(row);
    kinds.push(PointKind.CLUSTER_B);
  }

  return { X, kinds, nPerCluster };
}

function sampleGaussian(dim, scale, rng, mean) {
  const row = new Float64Array(dim);
  const sd = Math.sqrt(scale);
  for (let d = 0; d < dim; d++) {
    row[d] = (mean ? mean[d] : 0) + sd * randn(rng);
  }
  return row;
}

/** Center rows in place (returns same array). */
export function centerInPlace(X) {
  const n = X.length;
  if (n === 0) return X;
  const dim = X[0].length;
  const mu = new Float64Array(dim);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) mu[d] += X[i][d];
  }
  for (let d = 0; d < dim; d++) mu[d] /= n;
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < dim; d++) X[i][d] -= mu[d];
  }
  return X;
}

/** Append outliers: alpha * N(0, I) per utils.append_outliers */
export function appendOutliers(X, kinds, numOutliers, alpha, rng) {
  const dim = X[0].length;
  for (let i = 0; i < numOutliers; i++) {
    const row = new Float64Array(dim);
    for (let d = 0; d < dim; d++) row[d] = alpha * randn(rng);
    X.push(row);
    kinds.push(PointKind.OUTLIER);
  }
  return { X, kinds };
}

/** Poison: one point = global mean; more = random neighborhood means (utils.add_poison). */
export function addPoison(X, kinds, numPoisons, neighborhoodSize, rng) {
  const dim = X[0].length;
  const n = X.length;
  const k = Math.min(neighborhoodSize, n);
  for (let p = 0; p < numPoisons; p++) {
    const row = new Float64Array(dim);
    if (numPoisons === 1) {
      for (let i = 0; i < n; i++) {
        for (let d = 0; d < dim; d++) row[d] += X[i][d];
      }
      for (let d = 0; d < dim; d++) row[d] /= n;
    } else {
      const idx = randomSubset(n, k, rng);
      for (const i of idx) {
        for (let d = 0; d < dim; d++) row[d] += X[i][d];
      }
      for (let d = 0; d < dim; d++) row[d] /= k;
    }
    X.push(row);
    kinds.push(PointKind.POISON);
  }
  return { X, kinds };
}

function randomSubset(n, k, rng) {
  const idx = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx.slice(0, k);
}

export function buildDataset(params, rng) {
  let { X, kinds } = twoGaussians({
    nPerCluster: params.nPerCluster,
    distance: params.distance,
    dim: params.dim,
    power: params.power,
    c: params.c,
    rng,
  });
  centerInPlace(X);

  if (params.numOutliers > 0) {
    ({ X, kinds } = appendOutliers(X, kinds, params.numOutliers, params.outlierAlpha, rng));
  }
  if (params.numPoison > 0) {
    ({ X, kinds } = addPoison(
      X,
      kinds,
      params.numPoison,
      params.poisonNeighborhood,
      rng
    ));
  }
  return { X, kinds };
}

/** Cluster points only (no outliers or poison). */
export function clustersOnly(X, kinds) {
  const Xc = [];
  const kc = [];
  for (let i = 0; i < kinds.length; i++) {
    if (kinds[i] === PointKind.CLUSTER_A || kinds[i] === PointKind.CLUSTER_B) {
      Xc.push(X[i]);
      kc.push(kinds[i]);
    }
  }
  return { X: Xc, kinds: kc };
}

export function buildDatasetForMode(mode, params, rng) {
  let { X, kinds } = twoGaussians({
    nPerCluster: params.nPerCluster,
    distance: params.distance,
    dim: params.dim,
    power: params.power,
    c: params.c,
    rng,
  });
  centerInPlace(X);

  if (mode === "outlier" && params.numOutliers > 0) {
    ({ X, kinds } = appendOutliers(
      X,
      kinds,
      params.numOutliers,
      params.outlierAlpha,
      rng
    ));
  } else if (mode === "poison" && params.numPoison > 0) {
    ({ X, kinds } = addPoison(
      X,
      kinds,
      params.numPoison,
      params.poisonNeighborhood,
      rng
    ));
  }
  return { X, kinds };
}
