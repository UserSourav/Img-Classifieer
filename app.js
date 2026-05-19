/**
 * V-Image — app.js
 * Classification flow:
 *   1. Try Google Vision API via /api/classify (Vercel serverless function)
 *   2. If API call fails for any reason, fall back to MobileNet v3 in-browser
 */

'use strict';

/* ─────────────────────────────────────────────
   CONFIG
───────────────────────────────────────────── */
const CONFIG = {
  TOP_K: 10,
  DISPLAY_K: 5,
  LOW_CONF_THRESHOLD: 0.40,
  INPUT_SIZE: 224,
  PRESAMPLE_SIZE: 256,
  // Set to false to always use MobileNet (useful for local dev without backend)
  USE_VISION_API: true,
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
let mobileNetModel = null;
let mobilenetReady = false;

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
   MOBILENET — loads in background as fallback
───────────────────────────────────────────── */
async function loadMobileNet() {
  if (!window.mobilenet) return;
  try {
    setBadge('loading', 'Loading fallback model…');
    mobileNetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
    mobilenetReady = true;
    setBadge('ready', 'Ready');
    console.log('[V-Image] MobileNet fallback ready');
  } catch (err) {
    console.warn('[V-Image] MobileNet failed to load:', err.message);
    setBadge('error', 'Model error');
  }
}

/* ─────────────────────────────────────────────
   IMAGE → BASE64
   Draws the image onto a canvas and extracts
   a base64 JPEG string to send to the API.
───────────────────────────────────────────── */
function imageToBase64(imgEl) {
  const canvas = document.createElement('canvas');

  // Cap at 1024px to keep payload size reasonable
  const MAX = 1024;
  let w = imgEl.naturalWidth  || imgEl.width;
  let h = imgEl.naturalHeight || imgEl.height;
  if (w > MAX || h > MAX) {
    const scale = MAX / Math.max(w, h);
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }

  canvas.width  = w;
  canvas.height = h;
  canvas.getContext('2d').drawImage(imgEl, 0, 0, w, h);

  // Returns base64 string WITHOUT the "data:image/jpeg;base64," prefix
  return canvas.toDataURL('image/jpeg', 0.88).split(',')[1];
}

/* ─────────────────────────────────────────────
   GOOGLE VISION API  (via Vercel serverless)
───────────────────────────────────────────── */
async function classifyWithVisionAPI(imgEl) {
  const imageBase64 = imageToBase64(imgEl);

  const res = await fetch('/api/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ imageBase64 }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  return await res.json();
}

/* ─────────────────────────────────────────────
   MOBILENET INFERENCE  (fallback)
───────────────────────────────────────────── */
async function classifyWithMobileNet(imgEl) {
  if (!mobilenetReady || !mobileNetModel) {
    throw new Error('MobileNet not ready');
  }
  const t0 = performance.now();
  const preds = await mobileNetModel.classify(imgEl, CONFIG.TOP_K);
  const elapsed = performance.now() - t0;
  return {
    source: 'mobilenet',
    elapsed,
    labels: preds.map(p => ({
      label: prettyLabel(p.className),
      confidence: parseFloat((p.probability * 100).toFixed(1)),
    })),
    objects: [],
  };
}

/* ─────────────────────────────────────────────
   MAIN CLASSIFY — tries Vision API, falls back
───────────────────────────────────────────── */
async function runClassify(imgEl) {
  setThinking('Analyzing image…');
  resultsDiv.classList.add('hidden');
  thinking.classList.remove('hidden');

  let result = null;

  if (CONFIG.USE_VISION_API) {
    try {
      const t0 = performance.now();
      result = await classifyWithVisionAPI(imgEl);
      result.elapsed = performance.now() - t0;
      console.log('[V-Image] Google Vision result:', result);
    } catch (err) {
      console.warn('[V-Image] Vision API failed, falling back to MobileNet:', err.message);
      setBadge('ready', 'API failed — using MobileNet');
    }
  }

  // Fallback to MobileNet if Vision API failed or is disabled
  if (!result) {
    try {
      result = await classifyWithMobileNet(imgEl);
    } catch (err) {
      thinking.innerHTML = `<span style="color:#ff6b6b">Error: ${err.message}</span>`;
      return;
    }
  }

  showResults(result);
}

/* ─────────────────────────────────────────────
   RENDER RESULTS
───────────────────────────────────────────── */
function showResults(result) {
  predList.innerHTML = '';

  const isVision    = result.source === 'google-vision';
  const isMobileNet = result.source === 'mobilenet';

  // Google Vision: prefer objects (more specific) over labels
  let items = [];
  if (isVision) {
    // Merge objects + labels, deduplicate by lowercase label
    const seen = new Set();
    const merged = [...(result.objects || []), ...(result.labels || [])];
    for (const item of merged) {
      const key = item.label.toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        items.push(item);
      }
    }
    items = items.slice(0, CONFIG.DISPLAY_K);
  } else {
    items = (result.labels || []).slice(0, CONFIG.DISPLAY_K);
  }

  if (items.length === 0) {
    predList.innerHTML = '<li style="color:var(--muted);font-size:0.78rem">No results returned.</li>';
  }

  const topConf = items[0]?.confidence || 0;
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

  // Low confidence warning (MobileNet only)
  let warningHTML = '';
  if (isMobileNet && isLowConf) {
    warningHTML = `
      <div class="low-conf-warning">
        Low confidence (${topConf.toFixed(0)}%) —
        try a clearer, well-lit photo of a single object.
      </div>`;
  }

  const sourceLabel = isVision ? 'Google Vision API' : 'MobileNet v3 (fallback)';
  const elapsedText = result.elapsed ? `${Math.round(result.elapsed)} ms` : '—';

  metaDiv.innerHTML = `
    ${warningHTML}
    <span>Model: ${sourceLabel}</span><br>
    <span>Inference: ${elapsedText}</span><br>
    ${topConf ? `<span>Top confidence: ${topConf.toFixed(1)}%</span>` : ''}
  `;

  thinking.classList.add('hidden');
  resultsDiv.classList.remove('hidden');
  setBadge('ready', isVision ? 'Google Vision ready' : 'Ready');
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
loadMobileNet();
