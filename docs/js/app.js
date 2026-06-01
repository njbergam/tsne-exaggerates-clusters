import { mulberry32 } from "./math.js";
import { pairwiseSqDist } from "./math.js";
import { buildDatasetForMode, clustersOnly } from "./data.js";
import { pca2D } from "./pca.js";
import { computeJointP, createTsneState, tsneStep } from "./tsne.js";
import { ScatterPlot, isInjection } from "./plot.js";
import {
  MatrixHeatmap,
  orderIndicesByKind,
  permuteSquareMatrix,
} from "./matrixHeatmap.js";

const $ = (id) => document.getElementById(id);

const Phase = {
  IDLE: "idle",
  TSNE: "tsne",
  CONVERGED: "converged",
};

const STATIONARY_FRAMES = 10;
const MOVE_TOL = 2e-3;
const MIN_ITER = 150;
const EXAGGERATION_END = 100;
const STEPS_PER_FRAME = 4;
const LEARNING_RATE = 200;
const MOMENTUM = 0.5;

const MODE_BLURBS = {
  outlier:
    "Add outliers α·𝒩(0,I). Compare t-SNE on full data vs clusters only.",
  poison:
    "One poison point = global mean; more use random neighborhood means. Compare t-SNE with vs without them.",
};

let mode = "outlier";
let phase = Phase.IDLE;
let rng = mulberry32(42);

/** Full dataset (with injection). */
let X = null;
let kinds = [];
let n = 0;
let pcaY = null;
let jointP = null;

/** Clusters only (no injection). */
let XBase = null;
let kindsBase = [];
let nBase = 0;
let pcaYBase = null;
let jointPBase = null;

function makeRun() {
  return { state: null, converged: false, stationaryCount: 0 };
}

const runWith = makeRun();
const runWithout = makeRun();

let tsnePaused = false;
let tsneSpeed = 1;

const pcaPlot = new ScatterPlot($("plotPca"));
const pMatrixPlot = new MatrixHeatmap($("plotP"));
const tsnePlotWith = new ScatterPlot($("plotTsneWith"));
const tsnePlotWithout = new ScatterPlot($("plotTsneWithout"));
const overlay = $("startOverlay");
const statusEl = $("status");
const statusWithEl = $("statusWith");
const statusWithoutEl = $("statusWithout");
const btnStart = $("btnStartTsne");
const btnPause = $("btnPause");

function readParams() {
  return {
    seed: Number($("seed").value) || 42,
    nPerCluster: Number($("nPerCluster").value),
    distance: Number($("distance").value),
    dim: Number($("dim").value),
    power: Number($("power").value),
    numOutliers: Number($("numOutliers").value),
    outlierAlpha: Number($("outlierAlpha").value),
    numPoison: Number($("numPoison").value),
    poisonNeighborhood: Number($("poisonNeighborhood").value),
    perplexity: Number($("perplexity").value),
  };
}

function effectivePerplexity(requested, sampleSize) {
  const cap = Math.max(5, Math.floor((sampleSize - 1) / 3));
  return Math.min(requested, cap);
}

function bindSlider(id, labelId, fmt = (v) => v) {
  const el = $(id);
  const lab = $(labelId);
  const update = () => {
    lab.textContent = fmt(el.value);
  };
  el.addEventListener("input", update);
  update();
}

function applyPcaHeight() {
  const px = Number($("pcaHeight").value);
  document.documentElement.style.setProperty("--pca-height", `${px}px`);
  $("pcaHeightVal").textContent = String(px);
  pcaPlot.resize();
  pMatrixPlot.resize();
}

function computeAffinityP(points, count) {
  const D = pairwiseSqDist(points);
  const perp = effectivePerplexity(readParams().perplexity, count);
  return computeJointP(D, count, perp);
}

function updatePMatrixView() {
  if (!jointP || !kinds.length) {
    pMatrixPlot.clear("Computing P…");
    return;
  }
  const order = orderIndicesByKind(kinds);
  const Psorted = permuteSquareMatrix(jointP, n, order);
  const sortedKinds = order.map((i) => kinds[i]);
  pMatrixPlot.setMatrix(Psorted, n, sortedKinds);
}

function injectionCount() {
  return kinds.filter((k) => isInjection(k)).length;
}

function updateTsneLabels() {
  const params = readParams();
  const inj = injectionCount();
  if (mode === "outlier") {
    const word = inj === 1 ? "outlier" : "outliers";
    $("tsneLabelWith").textContent =
      inj === 0 ? "With data (no outliers)" : `With ${inj} ${word}`;
  } else {
    const word = inj === 1 ? "poison point" : "poison points";
    $("tsneLabelWith").textContent =
      inj === 0 ? "With data (no poison)" : `With ${inj} ${word}`;
  }
}

function stepsPerFrame() {
  return STEPS_PER_FRAME * tsneSpeed;
}

function syncTsneToolbar() {
  const running = phase === Phase.TSNE;
  btnPause.classList.toggle("hidden", !running);
  btnPause.textContent = tsnePaused ? "Resume" : "Pause";
  btnPause.classList.toggle("paused", tsnePaused);
}

function setTsneSpeed(mult) {
  tsneSpeed = mult;
  document.querySelectorAll(".btn-speed").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.speed) === mult);
  });
}

function resetRuns() {
  runWith.state = null;
  runWith.converged = false;
  runWith.stationaryCount = 0;
  runWithout.state = null;
  runWithout.converged = false;
  runWithout.stationaryCount = 0;
}

function setPhase(next) {
  phase = next;
  overlay.classList.toggle("hidden", phase === Phase.TSNE);
  statusEl.classList.remove("converged");

  if (phase === Phase.IDLE) {
    tsnePaused = false;
    btnStart.textContent = "Start t-SNE";
    btnStart.disabled = false;
    tsnePlotWith.setEmptyMessage("Press Start t-SNE");
    tsnePlotWithout.setEmptyMessage("Press Start t-SNE");
    if (!runWith.state) {
      tsnePlotWith.setData(null, [], 0);
      tsnePlotWithout.setData(null, [], 0);
    }
    statusWithEl.textContent = "—";
    statusWithoutEl.textContent = "—";
  } else if (phase === Phase.TSNE) {
    tsnePaused = false;
    btnStart.disabled = true;
  } else if (phase === Phase.CONVERGED) {
    tsnePaused = false;
    statusEl.classList.add("converged");
    overlay.classList.remove("hidden");
    btnStart.textContent = "Run t-SNE again";
    btnStart.disabled = false;
  }
  syncTsneToolbar();
}

function setGlobalStatus(text) {
  statusEl.textContent = tsnePaused && phase === Phase.TSNE ? `${text} · paused` : text;
}

function maxMove(Y, Ybefore, len) {
  let max = 0;
  for (let i = 0; i < len; i++) {
    const d = Math.abs(Y[i] - Ybefore[i]);
    if (d > max) max = d;
  }
  return max;
}

function rebuildFromSettings() {
  const params = readParams();
  rng = mulberry32(params.seed);

  const data = buildDatasetForMode(mode, { ...params, c: 1.0 }, rng);
  X = data.X;
  kinds = data.kinds;
  n = X.length;

  const base = clustersOnly(X, kinds);
  XBase = base.X;
  kindsBase = base.kinds;
  nBase = XBase.length;

  pcaY = pca2D(X, rng);
  pcaYBase = pca2D(XBase, rng);
  jointP = computeAffinityP(X, n);
  jointPBase = computeAffinityP(XBase, nBase);

  pcaPlot.setData(pcaY, kinds, n);
  updatePMatrixView();
  updateTsneLabels();

  resetRuns();
  resetTsnePanel();
  setGlobalStatus("PCA & P updated · press Start t-SNE");
}

function resetTsnePanel() {
  resetRuns();
  setPhase(Phase.IDLE);
}

function prepareTsne() {
  if (!jointP) jointP = computeAffinityP(X, n);
  if (!jointPBase) jointPBase = computeAffinityP(XBase, nBase);

  runWith.state = createTsneState(jointP, n, rng, {
    initY: pcaY.slice(),
    initScale: 0.05,
    learningRate: LEARNING_RATE,
    momentum: MOMENTUM,
  });
  runWith.state.lastLoss = 0;
  runWith.converged = false;
  runWith.stationaryCount = 0;

  runWithout.state = createTsneState(jointPBase, nBase, rng, {
    initY: pcaYBase.slice(),
    initScale: 0.05,
    learningRate: LEARNING_RATE,
    momentum: MOMENTUM,
  });
  runWithout.state.lastLoss = 0;
  runWithout.converged = false;
  runWithout.stationaryCount = 0;

  tsnePlotWith.setData(runWith.state.Y, kinds, n);
  tsnePlotWithout.setData(runWithout.state.Y, kindsBase, nBase);

  statusWithEl.textContent = "Iteration 0";
  statusWithoutEl.textContent = "Iteration 0";
  setGlobalStatus("Iteration 0 · both panels");
}

function startTsne() {
  if (!pcaY || phase === Phase.TSNE) return;
  prepareTsne();
  tsnePaused = false;
  setPhase(Phase.TSNE);
}

function togglePause() {
  if (phase !== Phase.TSNE) return;
  tsnePaused = !tsnePaused;
  syncTsneToolbar();
  if (tsnePaused) {
    setGlobalStatus(
      `Paused · with: ${statusWithEl.textContent} · without: ${statusWithoutEl.textContent}`
    );
  }
}

function checkConvergence(movePerStep, iter) {
  if (iter < MIN_ITER) return false;
  if (iter < EXAGGERATION_END + 20) return false;
  return movePerStep < MOVE_TOL;
}

function advanceRun(run, plot, kindsRun, nRun, statusEl) {
  if (run.converged || !run.state) return;

  const steps = stepsPerFrame();
  const Ybefore = run.state.Y.slice();
  const { loss, iter } = tsneStep(run.state, {
    steps,
    learningRate: LEARNING_RATE,
    momentum: MOMENTUM,
    exaggerationEnd: EXAGGERATION_END,
  });

  plot.setData(run.state.Y, kindsRun, nRun);
  const movePerStep = maxMove(run.state.Y, Ybefore, nRun * 2) / steps;

  if (checkConvergence(movePerStep, iter)) {
    run.stationaryCount += 1;
    if (run.stationaryCount >= STATIONARY_FRAMES) {
      run.converged = true;
      statusEl.textContent = `Converged · iter ${iter} · KL ≈ ${loss.toFixed(4)}`;
      return;
    }
  } else {
    run.stationaryCount = 0;
  }

  statusEl.textContent = `Iteration ${iter} · KL ≈ ${loss.toFixed(4)}`;
}

function tsneFrame() {
  if (phase !== Phase.TSNE) return;

  advanceRun(runWith, tsnePlotWith, kinds, n, statusWithEl);
  advanceRun(runWithout, tsnePlotWithout, kindsBase, nBase, statusWithoutEl);

  if (runWith.converged && runWithout.converged) {
    setPhase(Phase.CONVERGED);
    setGlobalStatus("Both converged");
    return;
  }

  const w = runWith.converged ? "done" : statusWithEl.textContent;
  const wo = runWithout.converged ? "done" : statusWithoutEl.textContent;
  setGlobalStatus(`With: ${w} · Without: ${wo}`);
}

function animationLoop() {
  if (phase === Phase.TSNE && !tsnePaused) tsneFrame();
  requestAnimationFrame(animationLoop);
}

function setMode(next) {
  mode = next;
  document.querySelectorAll(".mode-tab").forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  $("panelOutlier").classList.toggle("hidden", mode !== "outlier");
  $("panelPoison").classList.toggle("hidden", mode !== "poison");
  $("modeBlurb").innerHTML = MODE_BLURBS[mode];
  $("injectionLabel").textContent = mode === "outlier" ? "Outliers" : "Poison";
  rebuildFromSettings();
}

function initControls() {
  bindSlider("nPerCluster", "nPerClusterVal");
  bindSlider("distance", "distanceVal", (v) => Number(v).toFixed(1));
  bindSlider("dim", "dimVal");
  bindSlider("power", "powerVal", (v) => Number(v).toFixed(2));
  bindSlider("numOutliers", "numOutliersVal");
  bindSlider("outlierAlpha", "outlierAlphaVal", (v) => Number(v).toFixed(1));
  bindSlider("numPoison", "numPoisonVal");
  bindSlider("poisonNeighborhood", "poisonNeighborhoodVal");
  bindSlider("perplexity", "perplexityVal");

  bindSlider("pcaHeight", "pcaHeightVal");
  $("pcaHeight").addEventListener("input", applyPcaHeight);
  applyPcaHeight();

  const debouncedRebuild = debounce(rebuildFromSettings, 300);
  document
    .querySelectorAll(
      "#controls input[type=range]:not(#pcaHeight), #controls input[type=number]"
    )
    .forEach((el) => {
      el.addEventListener("input", debouncedRebuild);
      el.addEventListener("change", debouncedRebuild);
    });

  document.querySelectorAll(".mode-tab").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  btnStart.addEventListener("click", startTsne);
  btnPause.addEventListener("click", togglePause);

  document.querySelectorAll(".btn-speed").forEach((btn) => {
    btn.addEventListener("click", () => setTsneSpeed(Number(btn.dataset.speed)));
  });
  setTsneSpeed(1);

  $("btnReseed").addEventListener("click", () => {
    $("seed").value = Math.floor(Math.random() * 1e6);
    rebuildFromSettings();
  });
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function resizePlots() {
  pcaPlot.resize();
  pMatrixPlot.resize();
  tsnePlotWith.resize();
  tsnePlotWithout.resize();
}

window.addEventListener("resize", resizePlots);
initControls();
setPhase(Phase.IDLE);
resizePlots();
rebuildFromSettings();
requestAnimationFrame(animationLoop);
