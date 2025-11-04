## Image Recognition usinf CIFAR-10 dataset built on CNN arch

Small image-recognition demo using a lightweight PyTorch CNN and Grad-CAM visualization, wrapped in a Streamlit UI.

This repository contains a simple CIFAR-10 classifier (PyTorch) and a Streamlit app (`app.py`) that:
- accepts an uploaded image,
- runs a forward pass through the model,
- draws a robust bounding box on the original image, and
- displays the predicted class and confidence.

## Contents

- `app.py` — Streamlit application (entrypoint).
- `model.py` — Defines `SimpleCNN` and `build_model()` used by the app.
- `gradcam.py` — Grad-CAM implementation used to generate class activation maps.
- `main.ipynb` — notebook used to create weights and config for the model.

