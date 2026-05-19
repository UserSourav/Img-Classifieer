# V-Image — Browser Image Classifier

A browser-based image classifier that identifies objects in photos using a neural network. No server, no uploads, no account required. Everything runs locally inside the user's browser tab.

Live demo: https://vimgcls.vercel.app

---

## How it works

When the page loads, the browser downloads the MobileNet v3 model (~16 MB) from jsDelivr's CDN and caches it locally. On repeat visits the model loads from cache instantly with no re-download.

When a user drops or selects an image, TensorFlow.js resizes it to 224x224 pixels, converts it to a numeric tensor, and runs it through the neural network using the device's GPU via WebGL. The model outputs probability scores across 1,000 ImageNet categories. The top 5 results are displayed with confidence percentages. The image never leaves the device.

---

## Features

- Drag and drop, file picker, or clipboard paste (Ctrl+V) to load images
- Classifies across 1,000 object categories trained on ImageNet
- Inference runs in 50 to 300 ms depending on device GPU
- Six built-in sample images to try immediately
- Fully responsive, works on mobile browsers
- Zero data sent to any server at any point

---

## Tech stack

| Component | Technology |
|---|---|
| Neural network engine | TensorFlow.js 4.17 |
| Classification model | MobileNet v3 Large |
| Model hosting | jsDelivr CDN |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Deployment | Vercel |

---

## Model details

- Architecture: MobileNet v3 Large
- Training dataset: ImageNet (1,000 classes)
- Input size: 224 x 224 pixels (images are auto-resized)
- Top-1 accuracy: approximately 75% on ImageNet benchmark
- Model size: approximately 16 MB
- License: Apache 2.0 (Google)

The model works best on clear, well-lit photos of a single object centered in the frame. Complex scenes, unusual angles, or objects outside the 1,000 ImageNet categories will produce lower confidence scores.

---

## Project structure

```
Img-Classifieer/
├── index.html      Main page, UI layout, styles, background canvas animation
├── app.js          Model loading, image preprocessing, inference, result rendering
└── style.css       Additional styles
```

---

## Running locally

No build step or package installation required. Open `index.html` directly in a browser, or serve it with any static file server.

```bash
# Using Python
python -m http.server 8080

# Using Node.js
npx serve .
```

Then open `http://localhost:8080` in your browser.

---

## Accuracy notes

MobileNet v3 is a lightweight model designed for speed on mobile devices. Its 75% top-1 accuracy is a hard ceiling for this architecture. For higher accuracy the options are:

- Switch to EfficientNet-Lite4 (~87% top-1, still browser-runnable)
- Use a server-side model via an external vision API

The current `app.js` in this repository includes an improved version that attempts to load EfficientNet-Lite4 first and falls back to MobileNet v3 if it fails, along with better image preprocessing and a label prettifier.

---

## Privacy

The MobileNet model file is downloaded once from jsDelivr's CDN servers on first visit and cached by the browser. After that, no external requests are made during classification. No image data, usage data, or analytics are collected or transmitted.

---

## Credits

- MobileNet v3 model by Google, licensed under Apache 2.0
- TensorFlow.js by Google, licensed under Apache 2.0
- Built by Sourav — ss6015@srmist.edu.in
- LinkedIn: linkedin.com/in/usersourav
- GitHub: github.com/Usersourav
