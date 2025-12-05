# C:\parktrack\carid\color_classifier.py

from __future__ import annotations

import torch
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List

import open_clip


# ----------------------------
# Global model + prompts setup
# ----------------------------

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# You can change model name if your embedder uses a different one
_MODEL_NAME = "ViT-B-32"
_PRETRAINED_NAME = "laion2b_s34b_b79k"

# Load OpenCLIP model + preprocess only once at import time
_model, _, _preprocess = open_clip.create_model_and_transforms(
    _MODEL_NAME,
    pretrained=_PRETRAINED_NAME,
)
_model = _model.to(_DEVICE)
_model.eval()

_tokenizer = open_clip.get_tokenizer(_MODEL_NAME)

# Define color classes and prompts
_COLOR_LABELS: List[str] = [
    "red",
    "blue",
    "white",
    "black",
    "silver",
    "gray",
    "green",
    "yellow",
]

_COLOR_PROMPTS: List[str] = [
    f"a photo of a {_color} car" for _color in _COLOR_LABELS
]

# Encode prompts once
with torch.no_grad():
    _text_tokens = _tokenizer(_COLOR_PROMPTS).to(_DEVICE)
    _text_embeds = _model.encode_text(_text_tokens)
    _text_embeds = _text_embeds / _text_embeds.norm(dim=-1, keepdim=True)


def _preprocess_crop_bgr(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert OpenCV BGR crop → RGB PIL → CLIP preprocess → torch Tensor [1,3,H,W].
    """
    if crop_bgr is None or crop_bgr.size == 0:
        raise ValueError("Empty crop passed to color classifier")

    # BGR (OpenCV) -> RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    img_t = _preprocess(pil_img).unsqueeze(0).to(_DEVICE)
    return img_t


def predict_car_color(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Predict dominant car color using OpenCLIP zero-shot classification.

    Args:
        crop_bgr: numpy array [H, W, 3] in BGR (OpenCV format)

    Returns:
        (color_label, confidence_prob) where:
            color_label: one of _COLOR_LABELS
            confidence_prob: float in [0, 1], softmax probability
    """
    img_t = _preprocess_crop_bgr(crop_bgr)

    with torch.no_grad():
        image_embed = _model.encode_image(img_t)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

        # cosine similarity: [1, D] x [N, D]^T -> [1, N]
        logits = (image_embed @ _text_embeds.T).squeeze(0)  # [N]

        # softmax over colors for probability-like score
        probs = logits.softmax(dim=0)
        best_idx = int(torch.argmax(probs).item())
        best_color = _COLOR_LABELS[best_idx]
        best_conf = float(probs[best_idx].item())

    return best_color, best_conf
