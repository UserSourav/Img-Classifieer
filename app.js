/**
 * V-Image — app.js
 * Classification using MediaPipe EfficientNet-Lite0
 * Runs entirely in-browser, no server required.
 */

'use strict';

/* ─────────────────────────────────────────────
   CONFIG
───────────────────────────────────────────── */
const CONFIG = {
  TOP_K: 10,
  DISPLAY_K: 5,
  LOW_CONF_THRESHOLD: 0.40,
};

/* ─────────────────────────────────────────────
   DOM REFS
───────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const dropZone     = $('dropZone');
const dropInner    = $('dropInner');
const fileInput    = $('fileInput');
const previewState = $('previewState');
const previewImg   = $('previewImg');
const resetBtn     = $('resetBtn');
const thinking     = $('thinking');
const resultsDiv   = $('results');
const predList     = $('predictions');
const metaDiv      = $('meta');
const tryAnother   = $('tryAnother');
const modelBadge   = $('modelBadge');
const modelStatus  = $('modelStatus');
const sampleGrid   = $('sampleGrid');

/* ─────────────────────────────────────────────
   STATE
───────────────────────────────────────────── */
let imageClassifier = null;
let modelReady      = false;

/* ─────────────────────────────────────────────
   BADGE HELPER
───────────────────────────────────────────── */
function setBadge(state, text) {
  if (!modelBadge || !modelStatus) return;
  modelBadge.className = 'model-badge ' + state;
  modelStatus.textContent = text;
  const dot = modelBadge.querySelector('.dot');
  if (dot) dot.className = 'dot ' + state;
}

/* ─────────────────────────────────────────────
   LABEL PRETTIFIER
───────────────────────────────────────────── */
function prettyLabel(raw) {
  return raw
    .split(',')[0]
    .trim()
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .replace(/\bOf\b/g, 'of')
    .replace(/\bA\b/g,  'a')
    .replace(/\bThe\b/g,'the')
    .replace(/\bAnd\b/g,'and');
}

/* ─────────────────────────────────────────────
   LOAD EFFICIENTNET-LITE0  (via MediaPipe)
   ~25 MB model, ~80% top-1 on ImageNet
   (vs MobileNet v3 ~75%)
───────────────────────────────────────────── */
async function loadEfficientNet() {
  try {
    setBadge('loading', 'Loading EfficientNet…');

    const { ImageClassifier, FilesetResolver } = await import(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/vision_bundle.mjs'
    );

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm'
    );

    imageClassifier = await ImageClassifier.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite',
        delegate: 'GPU',
      },
      maxResults: CONFIG.TOP_K,
      scoreThreshold: 0.0,
    });

    modelReady = true;
    setBadge('ready', 'EfficientNet ready');
    console.log('[V-Image] EfficientNet-Lite0 loaded');
  } catch (err) {
    console.error('[V-Image] EfficientNet failed to load:', err);
    setBadge('error', 'Model error');
  }
}

/* ─────────────────────────────────────────────
   CLASSIFY WITH EFFICIENTNET
───────────────────────────────────────────── */
async function classifyWithEfficientNet(imgEl) {
  if (!modelReady || !imageClassifier) {
    throw new Error('EfficientNet not ready');
  }

  const t0 = performance.now();
  const result = imageClassifier.classify(imgEl);
  const elapsed = performance.now() - t0;

  const classifications = result?.classifications?.[0]?.categories || [];

  return {
    source: 'efficientnet-lite0',
    elapsed,
    labels: classifications.map(c => ({
      label: prettyLabel(c.categoryName),
      confidence: parseFloat((c.score * 100).toFixed(1)),
    })),
  };
}

/* ─────────────────────────────────────────────
   MAIN CLASSIFY
───────────────────────────────────────────── */
async function runClassify(imgEl) {
  setThinking('Analyzing image…');
  resultsDiv.classList.add('hidden');
  thinking.classList.remove('hidden');

  let result = null;

  try {
    result = await classifyWithEfficientNet(imgEl);
  } catch (err) {
    thinking.innerHTML = `<span style="color:#ff6b6b">Error: ${err.message}</span>`;
    return;
  }

  showResults(result);
}

/* ─────────────────────────────────────────────
   RENDER RESULTS
───────────────────────────────────────────── */
function showResults(result) {
  predList.innerHTML = '';

  const items = (result.labels || []).slice(0, CONFIG.DISPLAY_K);

  if (items.length === 0) {
    predList.innerHTML = '<li style="color:var(--muted);font-size:0.78rem">No results returned.</li>';
  }

  const topConf  = items[0]?.confidence || 0;
  const isLowConf = topConf < (CONFIG.LOW_CONF_THRESHOLD * 100);

  items.forEach((item, i) => {
    const pct      = typeof item.confidence === 'number' ? item.confidence.toFixed(1) : '—';
    const barWidth = typeof item.confidence === 'number' ? item.confidence : 0;

    const li = document.createElement('li');
    li.className = 'prediction-item';
    li.innerHTML = `
      <div class="pred-header">
        <span class="pred-label" title="${item.label}">${item.label}</span>
        <span class="pred-pct">${pct}%</span>
      </div>
      <div class="pred-bar-track">
        <div class="pred-bar ${i === 0 ? 'top' : ''}" style="width:0%"></div>
      </div>`;
    predList.appendChild(li);

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        li.querySelector('.pred-bar').style.width = barWidth + '%';
      });
    });
  });

  let warningHTML = '';
  if (isLowConf) {
    warningHTML = `
      <div class="low-conf-warning">
        Low confidence (${topConf.toFixed(0)}%) —
        try a clearer, well-lit photo of a single object.
      </div>`;
  }

  const elapsedText = result.elapsed ? `${Math.round(result.elapsed)} ms` : '—';

  metaDiv.innerHTML = `
    ${warningHTML}
    <span>Model: EfficientNet-Lite0</span><br>
    <span>Inference: ${elapsedText}</span><br>
    ${topConf ? `<span>Top confidence: ${topConf.toFixed(1)}%</span>` : ''}
  `;

  thinking.classList.add('hidden');
  resultsDiv.classList.remove('hidden');
  setBadge('ready', 'EfficientNet ready');
}

/* ─────────────────────────────────────────────
   UI HELPERS
───────────────────────────────────────────── */
function setThinking(text) {
  thinking.innerHTML = `<span class="spinner"></span><span>${text}</span>`;
}

function resetUI() {
  previewImg.src = '';
  previewImg.crossOrigin = '';
  previewState.classList.add('hidden');
  dropInner.classList.remove('hidden');
  resultsDiv.classList.add('hidden');
  thinking.classList.remove('hidden');
  setThinking('Analyzing image…');
  predList.innerHTML = '';
  metaDiv.innerHTML  = '';
  fileInput.value    = '';
}

function displayImageAndClassify(src) {
  previewImg.crossOrigin = 'anonymous';
  previewImg.onload = () => {
    dropInner.classList.add('hidden');
    previewState.classList.remove('hidden');
    runClassify(previewImg);
  };
  previewImg.onerror = () => {
    alert('Could not load image. CORS may be blocking the URL.');
  };
  previewImg.src = src;
}

function loadImageFromFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  displayImageAndClassify(URL.createObjectURL(file));
}

/* ─────────────────────────────────────────────
   EVENT LISTENERS
───────────────────────────────────────────── */
fileInput.addEventListener('change', e => loadImageFromFile(e.target.files[0]));
resetBtn.addEventListener('click', resetUI);
tryAnother.addEventListener('click', resetUI);

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) loadImageFromFile(file);
});

document.addEventListener('paste', e => {
  const items = (e.clipboardData || e.originalEvent.clipboardData).items;
  for (const item of items) {
    if (item.kind === 'file' && item.type.startsWith('image/')) {
      loadImageFromFile(item.getAsFile());
      break;
    }
  }
});

if (sampleGrid) {
  sampleGrid.addEventListener('click', e => {
    const btn = e.target.closest('[data-url]');
    if (btn) displayImageAndClassify(btn.dataset.url);
  });
}

/* ─────────────────────────────────────────────
   EXTRA STYLES
───────────────────────────────────────────── */
(function injectStyles() {
  const style = document.createElement('style');
  style.textContent = `
    .low-conf-warning {
      background: rgba(255,160,0,0.08);
      border: 1px solid rgba(255,160,0,0.25);
      border-radius: 8px;
      padding: 0.6rem 0.8rem;
      font-size: 0.7rem;
      color: #ffb347;
      line-height: 1.6;
      margin-bottom: 0.85rem;
    }
  `;
  document.head.appendChild(style);
})();

/* ─────────────────────────────────────────────
   BOOT
───────────────────────────────────────────── */
setBadge('loading', 'Loading…');
loadEfficientNet();
