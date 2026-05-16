/* ===== V-Image — App Logic ===== */

const dropZone    = document.getElementById('dropZone');
const dropInner   = document.getElementById('dropInner');
const previewState= document.getElementById('previewState');
const fileInput   = document.getElementById('fileInput');
const previewImg  = document.getElementById('previewImg');
const resetBtn    = document.getElementById('resetBtn');
const tryAnother  = document.getElementById('tryAnother');
const thinking    = document.getElementById('thinking');
const results     = document.getElementById('results');
const predictions = document.getElementById('predictions');
const metaEl      = document.getElementById('meta');
const modelBadge  = document.getElementById('modelBadge');
const modelStatus = document.getElementById('modelStatus');
const dot         = modelBadge.querySelector('.dot');
const sampleGrid  = document.getElementById('sampleGrid');

let model = null;
let classifyStart = 0;

/* ── Load Model ── */
async function loadModel() {
  try {
    modelStatus.textContent = 'Loading model\u2026';
    model = await mobilenet.load({ version: 2, alpha: 1.0 });
    dot.className = 'dot ready';
    modelBadge.className = 'model-badge ready';
    modelStatus.textContent = 'MobileNet v3 ready';
  } catch (err) {
    dot.className = 'dot error';
    modelBadge.className = 'model-badge error';
    modelStatus.textContent = 'Model failed to load';
    console.error('Model load error:', err);
  }
}

/* ── Classify Image ── */
async function classify(imgEl) {
  if (!model) return;
  thinking.classList.remove('hidden');
  results.classList.add('hidden');
  predictions.innerHTML = '';
  metaEl.textContent = '';
  classifyStart = performance.now();

  try {
    const preds = await model.classify(imgEl, 5);
    const elapsed = ((performance.now() - classifyStart) / 1000).toFixed(2);
    renderResults(preds, elapsed);
  } catch (err) {
    thinking.classList.add('hidden');
    predictions.innerHTML = '<li style="color:#ff6b6b;font-size:0.8rem;">Classification failed. Try a different image.</li>';
    results.classList.remove('hidden');
    console.error('Classification error:', err);
  }
}

/* ── Render Results ── */
function renderResults(preds, elapsed) {
  thinking.classList.add('hidden');
  results.classList.remove('hidden');
  predictions.innerHTML = '';

  const maxProb = preds[0]?.probability || 1;

  preds.forEach((pred, i) => {
    const pct = (pred.probability * 100).toFixed(1);
    const barW = ((pred.probability / maxProb) * 100).toFixed(1);
    const label = cleanLabel(pred.className);

    const li = document.createElement('li');
    li.className = 'prediction-item';
    li.innerHTML = `
      <div class="pred-header">
        <span class="pred-label" title="${label}">${label}</span>
        <span class="pred-pct">${pct}%</span>
      </div>
      <div class="pred-bar-track">
        <div class="pred-bar ${i === 0 ? 'top' : ''}" data-width="${barW}"></div>
      </div>
    `;
    predictions.appendChild(li);
  });

  requestAnimationFrame(() => {
    document.querySelectorAll('.pred-bar').forEach(bar => {
      bar.style.width = bar.dataset.width + '%';
    });
  });

  metaEl.innerHTML =
    `Inference: ${elapsed}s&nbsp;&nbsp;|&nbsp;&nbsp;${preds.length} results&nbsp;&nbsp;|&nbsp;&nbsp;Processed locally`;
}

/* ── Clean Label ── */
function cleanLabel(raw) {
  const first = raw.split(',')[0].trim();
  return first.charAt(0).toUpperCase() + first.slice(1);
}

/* ── Show Preview ── */
function showPreview(src) {
  previewImg.src = src;
  dropInner.classList.add('hidden');
  previewState.classList.remove('hidden');
  thinking.classList.remove('hidden');
  results.classList.add('hidden');
  previewImg.onload = () => classify(previewImg);
}

/* ── Reset ── */
function reset() {
  previewImg.src = '';
  previewState.classList.add('hidden');
  dropInner.classList.remove('hidden');
  fileInput.value = '';
  predictions.innerHTML = '';
  metaEl.textContent = '';
}

/* ── File Input ── */
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => showPreview(ev.target.result);
  reader.readAsDataURL(file);
});

/* ── Drag & Drop ── */
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

['dragleave', 'dragend'].forEach(evt =>
  dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'))
);

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    const reader = new FileReader();
    reader.onload = (ev) => showPreview(ev.target.result);
    reader.readAsDataURL(file);
  }
});

/* ── Clipboard Paste ── */
document.addEventListener('paste', (e) => {
  const items = Array.from(e.clipboardData.items);
  const imageItem = items.find(i => i.type.startsWith('image/'));
  if (imageItem) {
    const file = imageItem.getAsFile();
    const reader = new FileReader();
    reader.onload = (ev) => showPreview(ev.target.result);
    reader.readAsDataURL(file);
  }
});

/* ── Sample Buttons ── */
sampleGrid.querySelectorAll('.sample-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const url = btn.dataset.url;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      showPreview(canvas.toDataURL('image/jpeg'));
    };
    img.onerror = () => showPreview(url);
    img.src = url;
  });
});

/* ── Reset & Try Another Buttons ── */
resetBtn.addEventListener('click', reset);
tryAnother.addEventListener('click', reset);

/* ── Init ── */
loadModel();
