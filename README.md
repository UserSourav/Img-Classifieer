# V-Image — Browser-Native Image Classifier

> Classify any image instantly using EfficientNet-Lite0 — runs entirely in your browser, no uploads, fully private.

---

## Overview

V-Image is a client-side image classification web app that runs a state-of-the-art neural network directly in the browser using WebAssembly and GPU acceleration. No image data ever leaves your device.

Drop an image, paste from clipboard, or pick a sample — and get top-5 predictions with confidence scores in under 300ms.

---

## Features

- **100% on-device inference** — no server, no uploads, no account
- **EfficientNet-Lite0** — ~80% top-1 accuracy on 1,000 ImageNet categories
- **Fast** — 50–300ms inference time depending on device GPU
- **Privacy-first** — zero data sent anywhere
- **Drag & drop, file browse, clipboard paste, or sample images**
- **Animated confidence bars** with low-confidence warnings
- **Responsive** — works on desktop and mobile
- **No cookies, no tracking**

---

## Demo

| Feature | Detail |
|---|---|
| Model | EfficientNet-Lite0 (ImageNet) |
| Output classes | 1,000 categories |
| Model size | ~25 MB (cached after first load) |
| Inference time | 50–300 ms |
| Top-1 accuracy | ~80% on ImageNet benchmark |
| Framework | MediaPipe Tasks Vision + WebAssembly |
| Supported formats | PNG, JPG, WEBP, GIF, BMP |
| Browser support | Chrome, Firefox, Safari, Edge (modern) |

---

## How It Works

1. **Model loads** — EfficientNet-Lite0 is fetched from Google's CDN and cached in your browser. Subsequent visits load instantly.
2. **Image is pre-processed** — resized and normalized locally for model input.
3. **Inference runs** — MediaPipe runs the model via WebAssembly with GPU delegate acceleration.
4. **Results shown** — top 5 predicted classes are ranked and displayed with animated confidence bars.

---

## Getting Started

V-Image is a single HTML file with no build step required.

### Run locally

```bash
# Clone the repo
git clone https://github.com/Usersourav/v-image.git
cd v-image

# Serve with any static file server, e.g.:
npx serve .
# or
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

> **Note:** The app must be served over HTTP/HTTPS (not opened as a `file://` URL) for WebAssembly and cross-origin model loading to work correctly.

### No dependencies to install

All runtime dependencies (MediaPipe, EfficientNet model) are loaded from CDN on first use. The model (~25 MB) is cached by your browser automatically.

---

## Usage

| Method | How |
|---|---|
| Drag & drop | Drop any image onto the drop zone |
| File browser | Click **Browse files** |
| Clipboard | Press `Ctrl+V` / `Cmd+V` to paste a copied image |
| Sample images | Click any sample button (Dog, Cat, Elephant, etc.) |

After classification, results show the top 5 predicted categories with confidence percentages. A warning is shown when top confidence is below 40%.

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML model | EfficientNet-Lite0 (TFLite, float32) |
| Inference runtime | [MediaPipe Tasks Vision](https://developers.google.com/mediapipe) |
| Acceleration | WebAssembly + GPU delegate |
| Model source | [Google MediaPipe Models CDN](https://storage.googleapis.com/mediapipe-models/) |
| Fonts | Syne, DM Mono (Google Fonts) |
| Background FX | Custom canvas animation (vanilla JS) |

---

## Project Structure

```
v-image/
└── index.html     # Entire app — markup, styles, canvas animation, and ML logic
```

---

## Limitations

- Works best on **clear, well-lit photos of single objects**
- Does **not** recognise faces, text, or document content
- Trained on ImageNet's 1,000 classes — objects outside this set will produce low-confidence results
- Older mobile devices may experience slower inference times

---

## FAQ

**Is my image uploaded anywhere?**
No. All processing happens locally in your browser tab. No image data is ever sent to a server.

**Why does it take a few seconds the first time?**
The browser downloads EfficientNet-Lite0 (~25 MB) from Google's CDN on the first visit. After that, it's cached and loads in under a second.

**Can I use this on mobile?**
Yes. Tap **Browse files** to select from your photo library or camera roll. Inference may be slightly slower on older devices.

---

## Author

**Sourav**
- Email: [ss6015@srmist.edu.in](mailto:ss6015@srmist.edu.in)
- LinkedIn: [linkedin.com/in/usersourav](https://linkedin.com/in/usersourav)
- GitHub: [github.com/Usersourav](https://github.com/Usersourav)

---

## License

This project is open source. Feel free to fork, modify, and build on it.

### Third-party attributions

| Component | License |
|---|---|
| **EfficientNet-Lite0** — Google LLC | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| **MediaPipe Tasks Vision** — Google LLC | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| **Syne & DM Mono fonts** — Google Fonts | [OFL 1.1](https://scripts.sil.org/OFL) |

EfficientNet-Lite0 is a neural network architecture developed by Google and trained on the [ImageNet](https://www.image-net.org/) dataset. The model weights are distributed by Google under the Apache 2.0 license via the [MediaPipe Models CDN](https://storage.googleapis.com/mediapipe-models/). Use of the model is subject to that license.
