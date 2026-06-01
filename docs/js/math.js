/** Seeded PRNG and linear algebra helpers. */

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randn(rng) {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function zeros(rows, cols) {
  const out = new Array(rows);
  for (let i = 0; i < rows; i++) {
    out[i] = new Float64Array(cols);
  }
  return out;
}

export function pairwiseSqDist(X) {
  const n = X.length;
  const D = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    D[i * n + i] = 0;
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      const xi = X[i];
      const xj = X[j];
      for (let d = 0; d < xi.length; d++) {
        const t = xi[d] - xj[d];
        s += t * t;
      }
      D[i * n + j] = s;
      D[j * n + i] = s;
    }
  }
  return D;
}

export function maxOf(arr) {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) m = Math.max(m, arr[i]);
  return m;
}
