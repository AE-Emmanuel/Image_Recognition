import streamlit as st, torch, json
from PIL import Image, ImageDraw
from torchvision import transforms
import torch.nn.functional as F
from model import build_model
from gradcam import GradCAM
import torch.nn as nn


def get_last_conv_layer(model: nn.Module) -> nn.Module:
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found in the model.")
    return last
st.title("CIFAR-10 Image Classifier 🔎")

model=build_model()
model.load_state_dict(torch.load("weights/model.pth",map_location="cpu"))
model.eval()

classes=json.load(open("configs/classes.json"))
norm=json.load(open("configs/norm.json"))
normalize=transforms.Normalize(norm["mean"],norm["std"])
preprocess=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),normalize])

file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file).convert("RGB")

    # --- Preprocess ---
    x = preprocess(img).unsqueeze(0)  # [1,3,32,32]

    # --- GradCAM setup ---
    target_layer = get_last_conv_layer(model)   # or explicit: model.block2[0] / model.block3[0]
    cam_engine = GradCAM(model, target_layer)

    # --- Forward & Backward passes ---
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        x = x.requires_grad_(True)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1))
        score = logits[:, pred]
        score.backward(retain_graph=True)

    # --- Generate CAM (Hc, Wc depend on your model) ---
    cam = cam_engine.generate()[0].detach().cpu().numpy()

    # ============== Robust bbox on original image ==============
    import numpy as np, cv2
    from PIL import ImageDraw

    H, W = img.height, img.width

    # 1) Resize & smooth
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
    cam_blur = cv2.GaussianBlur(cam_resized, (0, 0), sigmaX=7, sigmaY=7)

    def find_bbox_from_cam(cam_blur, W, H, min_area_ratio=0.01):
        """Return (x, y, w, h) or None."""
        img_area = W * H
        for p in range(90, 49, -5):  # 90,85,80,...,50
            thr = np.percentile(cam_blur, p)
            mask = (cam_blur >= thr).astype(np.uint8) * 255
            # Dilate to grow region
            k = max(3, int(0.02 * min(W, H)))
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area_ratio * img_area:
                return x, y, w, h
        return None  # nothing decent found

    bbox = find_bbox_from_cam(cam_blur, W, H, min_area_ratio=0.01)

    # 3) Draw on a PIL copy of the original
    img_out = img.copy()
    draw = ImageDraw.Draw(img_out)

    if bbox is None:
        # Fallback: whole image with padding inside bounds
        pad = int(0.03 * min(W, H))
        x1, y1, x2, y2 = pad, pad, W - pad, H - pad
    else:
        x, y, w, h = bbox
        pad = int(0.05 * min(W, H))  # 5% padding
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(W, x + w + pad)
        y2 = min(H, y + h + pad)

    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

    # 4) Show ONLY the result image (no original upload)
    st.image(
        img_out,
        caption=f"Prediction: {classes[pred]} ({probs[0, pred].item()*100:.2f}%)",
        use_container_width=True
    )