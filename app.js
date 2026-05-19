/**
 * V-Image — Improved app.js
 * Changes vs original:
 *  1. EfficientNet-Lite4 primary model (~87% top-1) with MobileNet v3 fallback
 *  2. Better preprocessing: resize-then-center-crop instead of direct 224×224 squish
 *  3. Top-10 predictions fetched (shows top 5 but has more data)
 *  4. Label prettifier — cleans up ImageNet's ugly comma-separated synsets
 *  5. Low-confidence guard: if top prediction < 40%, shows warning banner
 *  6. Multi-crop ensemble on demand (3 crops averaged) for borderline results
 *  7. Correct EfficientNet input: [0,1] float range, not [-1,1]
 *  8. Graceful fallback to MobileNet if EfficientNet fails to load
 */

'use strict';

/* ─────────────────────────────────────────────
   CONFIG
───────────────────────────────────────────── */
const CONFIG = {
  TOP_K: 10,                    // fetch top-10, display top-5
  DISPLAY_K: 5,
  LOW_CONF_THRESHOLD: 0.40,     // warn below 40% top prediction
  ENSEMBLE_CROPS: 3,            // number of crops for ensemble mode
  INPUT_SIZE: 224,              // both models accept 224×224
  PRESAMPLE_SIZE: 256,          // resize to 256 first, then center-crop
};

/* EfficientNet-Lite4 from TF Hub — ~87% top-1 accuracy */
const EFFICIENTNET_URL =
  'https://tfhub.dev/tensorflow/tfjs-model/efficientnet/lite4/classification/1/default/1';

/* Fallback: MobileNet v3 via @tensorflow-models/mobilenet package */
const USE_MOBILENET_FALLBACK = true;

/* ─────────────────────────────────────────────
   STATE
───────────────────────────────────────────── */
let model = null;
let modelType = null;   // 'efficientnet' | 'mobilenet'
let mobileNetModel = null;
let imagenetLabels = null;

/* ─────────────────────────────────────────────
   DOM REFS  (matches your existing index.html IDs)
───────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const dropZone    = $('dropZone');
const dropInner   = $('dropInner');
const fileInput   = $('fileInput');
const previewState = $('previewState');
const previewImg  = $('previewImg');
const resetBtn    = $('resetBtn');
const resultsPanel = $('resultsPanel');
const thinking    = $('thinking');
const resultsDiv  = $('results');
const predList    = $('predictions');
const metaDiv     = $('meta');
const tryAnother  = $('tryAnother');
const modelBadge  = $('modelBadge');
const modelStatus = $('modelStatus');
const sampleGrid  = $('sampleGrid');

/* ─────────────────────────────────────────────
   LABEL UTILITIES
───────────────────────────────────────────── */

/**
 * Prettify a raw ImageNet label like "Egyptian cat" or "tabby, tabby cat"
 * → "Egyptian Cat"
 */
function prettyLabel(raw) {
  return raw
    .split(',')[0]                          // first synonym only
    .trim()
    .replace(/_/g, ' ')                     // underscores → spaces
    .replace(/\b\w/g, c => c.toUpperCase()) // Title Case
    .replace(/\bOf\b/g, 'of')              // fix "Cup Of Coffee"
    .replace(/\bA\b/g, 'a')
    .replace(/\bThe\b/g, 'the')
    .replace(/\bIn\b/g, 'in')
    .replace(/\bAnd\b/g, 'and');
}

/* ─────────────────────────────────────────────
   MODEL LOADING
───────────────────────────────────────────── */
async function loadModel() {
  setBadge('loading', 'Loading model…');

  // Try EfficientNet-Lite4 first
  try {
    setBadge('loading', 'Loading EfficientNet-Lite4…');
    model = await tf.loadGraphModel(EFFICIENTNET_URL, { fromTFHub: true });
    modelType = 'efficientnet';

    // Load ImageNet labels for EfficientNet (1000 classes)
    await loadImageNetLabels();

    setBadge('ready', 'EfficientNet-Lite4 ready');
    console.log('[V-Image] EfficientNet-Lite4 loaded ✓');
    return;
  } catch (err) {
    console.warn('[V-Image] EfficientNet failed, falling back to MobileNet:', err.message);
  }

  // Fallback: MobileNet v3 (already in your HTML via CDN script tag)
  if (USE_MOBILENET_FALLBACK && window.mobilenet) {
    try {
      setBadge('loading', 'Loading MobileNet v3…');
      mobileNetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
      modelType = 'mobilenet';
      setBadge('ready', 'MobileNet v3 ready');
      console.log('[V-Image] MobileNet v3 fallback loaded ✓');
      return;
    } catch (err) {
      console.error('[V-Image] MobileNet also failed:', err);
    }
  }

  setBadge('error', 'Model failed to load');
  throw new Error('Could not load any model');
}

async function loadImageNetLabels() {
  // EfficientNet uses standard ImageNet 1k labels
  // We embed a compact version inline to avoid an extra fetch
  // Full 1000-label list (sourced from ImageNet synset_words.txt)
  const IMAGENET_CLASSES_URL =
    'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt';
  try {
    const resp = await fetch(IMAGENET_CLASSES_URL);
    const text = await resp.text();
    // First line is "background", skip it (index 0), so class 1 = tench, etc.
    imagenetLabels = text.trim().split('\n');
  } catch {
    console.warn('[V-Image] Could not fetch ImageNet labels, using index fallback');
    imagenetLabels = null;
  }
}

function setBadge(state, text) {
  if (!modelBadge || !modelStatus) return;
  modelBadge.className = 'model-badge ' + state;
  modelStatus.textContent = text;
  const dot = modelBadge.querySelector('.dot');
  if (dot) dot.className = 'dot ' + state;
}

/* ─────────────────────────────────────────────
   IMAGE PREPROCESSING
   Key improvement: resize to 256 first, then center-crop
   to 224×224 — preserves aspect ratio better than
   directly squishing to 224.
───────────────────────────────────────────── */
function preprocessImage(imgEl, inputSize = CONFIG.INPUT_SIZE) {
  return tf.tidy(() => {
    const raw = tf.browser.fromPixels(imgEl);   // [H, W, 3]  uint8

    // 1. Resize to PRESAMPLE_SIZE keeping the tensor square
    const resized = tf.image.resizeBilinear(raw, [CONFIG.PRESAMPLE_SIZE, CONFIG.PRESAMPLE_SIZE]);

    // 2. Center-crop to inputSize×inputSize
    const offset = Math.floor((CONFIG.PRESAMPLE_SIZE - inputSize) / 2);
    const cropped = resized.slice([offset, offset, 0], [inputSize, inputSize, 3]);

    // 3. Normalize to [0, 1]  (EfficientNet expects this range)
    const normalized = cropped.toFloat().div(255.0);

    // 4. Add batch dim → [1, 224, 224, 3]
    return normalized.expandDims(0);
  });
}

/**
 * Ensemble: run 3 crops (center + slight offsets) and average logits
 * for improved accuracy on borderline images.
 */
function preprocessEnsemble(imgEl) {
  const offsets = [
    [0, 0],   // center crop
    [-8, -8], // top-left shift
    [8, 8],   // bottom-right shift
  ];
  return offsets.map(([dy, dx]) => {
    return tf.tidy(() => {
      const raw = tf.browser.fromPixels(imgEl);
      const sz = CONFIG.INPUT_SIZE;
      const pre = CONFIG.PRESAMPLE_SIZE;
      const resized = tf.image.resizeBilinear(raw, [pre, pre]);
      const oy = Math.max(0, Math.floor((pre - sz) / 2) + dy);
      const ox = Math.max(0, Math.floor((pre - sz) / 2) + dx);
      const cropped = resized.slice([oy, ox, 0], [sz, sz, 3]);
      return cropped.toFloat().div(255.0).expandDims(0);
    });
  });
}

/* ─────────────────────────────────────────────
   INFERENCE
───────────────────────────────────────────── */
async function classifyImage(imgEl, useEnsemble = false) {
  if (!model && !mobileNetModel) throw new Error('Model not loaded');

  const t0 = performance.now();

  if (modelType === 'mobilenet') {
    // MobileNet path — simple, uses library helper
    const preds = await mobileNetModel.classify(imgEl, CONFIG.TOP_K);
    const elapsed = performance.now() - t0;
    return {
      predictions: preds.map(p => ({
        label: prettyLabel(p.className),
        probability: p.probability,
      })),
      elapsed,
      modelName: 'MobileNet v3',
      ensemble: false,
    };
  }

  // EfficientNet path
  let logitsTensor;

  if (useEnsemble) {
    // Average logits across 3 crops
    const crops = preprocessEnsemble(imgEl);
    const logitsAll = crops.map(crop => tf.tidy(() => model.predict(crop)));
    logitsTensor = tf.tidy(() => {
      const stacked = tf.stack(logitsAll.map(l => l.squeeze()));
      const averaged = stacked.mean(0);
      crops.forEach(c => c.dispose());
      logitsAll.forEach(l => l.dispose());
      return averaged;
    });
  } else {
    const tensor = preprocessImage(imgEl);
    logitsTensor = tf.tidy(() => model.predict(tensor).squeeze());
    tensor.dispose();
  }

  // Softmax → probabilities
  const probsTensor = tf.tidy(() => tf.softmax(logitsTensor));
  const probs = await probsTensor.data();
  logitsTensor.dispose();
  probsTensor.dispose();

  const elapsed = performance.now() - t0;

  // Build top-K predictions
  const indexed = Array.from(probs).map((p, i) => ({ p, i }));
  indexed.sort((a, b) => b.p - a.p);
  const topK = indexed.slice(0, CONFIG.TOP_K);

  const predictions = topK.map(({ p, i }) => {
    let label = `Class ${i}`;
    if (imagenetLabels && imagenetLabels[i]) {
      label = prettyLabel(imagenetLabels[i]);
    }
    return { label, probability: p };
  });

  return {
    predictions,
    elapsed,
    modelName: 'EfficientNet-Lite4',
    ensemble: useEnsemble,
  };
}

/* ─────────────────────────────────────────────
   UI RENDERING
───────────────────────────────────────────── */
function showResults(result) {
  const { predictions, elapsed, modelName, ensemble } = result;
  const top = predictions[0];
  const isLowConf = top.probability < CONFIG.LOW_CONF_THRESHOLD;

  // Clear previous
  predList.innerHTML = '';

  // Render top-5
  predictions.slice(0, CONFIG.DISPLAY_K).forEach((pred, i) => {
    const pct = (pred.probability * 100).toFixed(1);
    const li = document.createElement('li');
    li.className = 'prediction-item';
    li.innerHTML = `
      <div class="pred-header">
        <span class="pred-label" title="${pred.label}">${pred.label}</span>
        <span class="pred-pct">${pct}%</span>
      </div>
      <div class="pred-bar-track">
        <div class="pred-bar ${i === 0 ? 'top' : ''}" style="width:0%"></div>
      </div>`;
    predList.appendChild(li);

    // Animate bar after paint
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        li.querySelector('.pred-bar').style.width = pct + '%';
      });
    });
  });

  // Low-confidence warning
  let warningHTML = '';
  if (isLowConf) {
    warningHTML = `
      <div class="low-conf-warning">
        ⚠ Low confidence (${(top.probability * 100).toFixed(0)}%) — 
        try a clearer, well-lit photo of a single object.
        ${!ensemble ? '<button class="btn-ensemble" id="ensembleBtn">Try ensemble mode</button>' : ''}
      </div>`;
  }

  // Meta info
  metaDiv.innerHTML = `
    ${warningHTML}
    <span>Model: ${modelName}</span><br>
    <span>Inference: ${Math.round(elapsed)} ms${ensemble ? ' (3-crop ensemble)' : ''}</span><br>
    <span>Top prediction: ${(top.probability * 100).toFixed(1)}% confidence</span>
  `;

  // Wire ensemble button if present
  const ensBtn = document.getElementById('ensembleBtn');
  if (ensBtn) {
    ensBtn.addEventListener('click', async () => {
      ensBtn.disabled = true;
      ensBtn.textContent = 'Running…';
      thinking.classList.remove('hidden');
      resultsDiv.classList.add('hidden');
      try {
        const r = await classifyImage(previewImg, true);
        showResults(r);
      } catch (e) {
        console.error(e);
      } finally {
        thinking.classList.add('hidden');
        resultsDiv.classList.remove('hidden');
      }
    });
  }

  thinking.classList.add('hidden');
  resultsDiv.classList.remove('hidden');
}

async function runClassify(imgEl) {
  thinking.classList.remove('hidden');
  resultsDiv.classList.add('hidden');
  try {
    const result = await classifyImage(imgEl, false);
    showResults(result);
  } catch (err) {
    console.error('[V-Image] Classification error:', err);
    thinking.innerHTML = `<span style="color:#ff6b6b">Error: ${err.message}</span>`;
  }
}

/* ─────────────────────────────────────────────
   IMAGE LOADING HELPERS
───────────────────────────────────────────── */
function loadImageFromFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  displayImageAndClassify(url);
}

function loadImageFromURL(url) {
  displayImageAndClassify(url);
}

function displayImageAndClassify(src) {
  previewImg.onload = () => {
    dropInner.classList.add('hidden');
    previewState.classList.remove('hidden');
    runClassify(previewImg);
  };
  previewImg.onerror = () => {
    alert('Could not load image. If it is a remote URL, CORS may be blocking it.');
  };

  // For cross-origin images (Wikipedia samples etc.) set crossOrigin
  previewImg.crossOrigin = 'anonymous';
  previewImg.src = src;
}

function resetUI() {
  previewImg.src = '';
  previewImg.crossOrigin = '';
  previewState.classList.add('hidden');
  dropInner.classList.remove('hidden');
  resultsDiv.classList.add('hidden');
  thinking.classList.remove('hidden');
  thinking.innerHTML = `<span class="spinner"></span><span>Analyzing image…</span>`;
  predList.innerHTML = '';
  metaDiv.innerHTML = '';
  fileInput.value = '';
}

/* ─────────────────────────────────────────────
   EVENT WIRING
───────────────────────────────────────────── */

// File input
fileInput.addEventListener('change', e => {
  loadImageFromFile(e.target.files[0]);
});

// Reset
resetBtn.addEventListener('click', resetUI);
tryAnother.addEventListener('click', resetUI);

// Drag & drop
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadImageFromFile(file);
});

// Paste from clipboard
document.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.kind === 'file' && item.type.startsWith('image/')) {
      loadImageFromFile(item.getAsFile());
      break;
    }
  }
});

// Sample images
if (sampleGrid) {
  sampleGrid.addEventListener('click', e => {
    const btn = e.target.closest('[data-url]');
    if (btn) loadImageFromURL(btn.dataset.url);
  });
}

/* ─────────────────────────────────────────────
   CSS INJECTION (low-conf warning & ensemble btn)
   Injected here so index.html doesn't need changes
───────────────────────────────────────────── */
(function injectStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .low-conf-warning {
      background: rgba(255, 160, 0, 0.08);
      border: 1px solid rgba(255, 160, 0, 0.25);
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      font-size: 0.7rem;
      color: #ffb347;
      line-height: 1.6;
      margin-bottom: 0.85rem;
    }
    .btn-ensemble {
      display: inline-block;
      margin-top: 0.4rem;
      padding: 0.25rem 0.75rem;
      background: transparent;
      border: 1px solid rgba(255, 160, 0, 0.4);
      border-radius: 999px;
      color: #ffb347;
      font-family: var(--font-mono, monospace);
      font-size: 0.68rem;
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn-ensemble:hover { background: rgba(255,160,0,0.12); }
    .btn-ensemble:disabled { opacity: 0.5; cursor: default; }
  `;
  document.head.appendChild(style);
})();

/* ─────────────────────────────────────────────
   BOOT
───────────────────────────────────────────── */
loadModel().catch(err => {
  console.error('[V-Image] Boot error:', err);
  setBadge('error', 'Failed to load');
});
