## Image Recognition — CIFAR-10 Streamlit Explorer

Small image-recognition demo using a lightweight PyTorch CNN and Grad-CAM visualization, wrapped in a Streamlit UI.

This repository contains a simple CIFAR-10 classifier (PyTorch) and a Streamlit app (`app.py`) that:
- accepts an uploaded image,
- runs a forward pass through the model,
- computes a Grad-CAM heatmap (using `gradcam.py`),
- draws a robust bounding box on the original image, and
- displays the predicted class and confidence.

## Contents

- `app.py` — Streamlit application (entrypoint).
- `model.py` — Defines `SimpleCNN` and `build_model()` used by the app.
- `gradcam.py` — Grad-CAM implementation used to generate class activation maps.
- `configs/` — small JSON config files used by the app:
  - `classes.json` — CIFAR-10 class names
  - `norm.json` — normalization mean & std for preprocessing
- `weights/` — place your trained model here as `weights/model.pth` (ignored by git).
- `main.ipynb` — notebook (experiments / exploration).
- `data/` — local dataset examples or test images (ignored by git).

## Requirements

- Python: per `pyproject.toml` this project declares `requires-python = ">=3.13"`.
- See `requirements.txt` for the pinned/common packages used by the project.

Key packages used by the code:
- torch, torchvision (PyTorch)
- streamlit
- opencv-python / opencv-python-headless
- pillow, numpy

Note about PyTorch/GPU: install an appropriate `torch` wheel for your CUDA version if you want GPU acceleration. The `requirements.txt` includes a generic `torch>=2.9.0` which will install a CPU-only package by default on many setups — consult the official PyTorch install guide for GPU builds.

## Quick start

1. Create and activate a virtual environment (macOS / zsh shown):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Ensure `weights/model.pth` exists. If you don't have a trained model, you can train one separately or copy a sample model into `weights/model.pth`.

4. Run the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser. Upload an image (png, jpg, jpeg) and the app will show the predicted CIFAR-10 class and a bounding box produced from the Grad-CAM heatmap.

## Running the notebook

Open `main.ipynb` with Jupyter or VS Code's notebook viewer. The notebook may contain exploratory code used during development.

## File structure and notes

- `app.py` expects the model weights at `weights/model.pth` and configuration JSONs in `configs/`.
- `model.py` contains a very small CNN (for CIFAR-10). If you change input resolution or model architecture, update `app.py` preprocessing accordingly.
- `gradcam.py` attaches hooks to the chosen convolutional layer and computes the CAM during backward.

## Troubleshooting

- OpenCV import errors: if `import cv2` fails, ensure you installed `opencv-python` (or `opencv-python-headless` for headless environments):

```bash
pip install opencv-python
```

- PyTorch mismatch / CUDA errors: install the appropriate `torch` wheel per the official docs: https://pytorch.org/get-started/locally/

- If Streamlit doesn't serve, ensure you used the virtual environment that has `streamlit` installed and call `streamlit run app.py` from the project root.

## Contributing & Next steps

- Add a training script to save `weights/model.pth` automatically.
- Add a small set of sample images in `data/` for quick local testing (those are git-ignored by default).
- Consider adding unit tests for `gradcam.generate()` and the model forward pass.

## License

This project does not include a license file. Add `LICENSE` if you want to specify reuse rules.

---
If you'd like, I can:
- pin exact versions in `requirements.txt`,
- add a `launch` script for Streamlit (or a `Procfile` for deployments), or
- create a minimal training script and an example `weights/model.pth` (small random weights for quick smoke test).
