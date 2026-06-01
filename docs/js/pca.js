/**
 * Top-2 PCA via power iteration on covariance of centered data.
 */

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function matVec(C, d, v) {
  const out = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += C[i * d + j] * v[j];
    out[i] = s;
  }
  return out;
}

/** Returns Float64Array length 2*n with (pc1, pc2) per row. */
export function pca2D(X, rng = Math.random) {
  const n = X.length;
  const d = X[0].length;
  const C = new Float64Array(d * d);
  const inv = 1 / Math.max(n - 1, 1);

  for (let i = 0; i < n; i++) {
    const xi = X[i];
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        C[a * d + b] += xi[a] * xi[b] * inv;
      }
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
  }

  let v1 = new Float64Array(d);
  for (let i = 0; i < d; i++) v1[i] = rng() - 0.5;
  v1 = normalize(v1);

  for (let it = 0; it < 80; it++) {
    const w = matVec(C, d, v1);
    v1 = normalize(w);
  }

  const lambda1 = dot(v1, matVec(C, d, v1));
  const C2 = C.slice();
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < d; b++) {
      C2[a * d + b] -= lambda1 * v1[a] * v1[b];
    }
  }

  let v2 = new Float64Array(d);
  for (let i = 0; i < d; i++) v2[i] = rng() - 0.5;
  v2 = orthogonalize(v2, v1);
  for (let it = 0; it < 80; it++) {
    const w = matVec(C2, d, v2);
    v2 = orthogonalize(normalize(w), v1);
  }

  const Y = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    Y[i * 2] = dot(X[i], v1);
    Y[i * 2 + 1] = dot(X[i], v2);
  }
  return Y;
}

function normalize(v) {
  const nrm = norm(v) || 1;
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / nrm;
  return out;
}

function orthogonalize(v, u) {
  const proj = dot(v, u);
  const out = new Float64Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] - proj * u[i];
  const nrm = norm(out);
  if (nrm < 1e-12) return normalize(v);
  for (let i = 0; i < v.length; i++) out[i] /= nrm;
  return out;
}
