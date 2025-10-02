import numpy as np
import torch
import open_clip
import cv2
from PIL import Image

# CPU-only
_DEVICE = torch.device("cpu")

# Lazy singletons
_MODEL = None
_PREPROC = None

def _load_model():
    global _MODEL, _PREPROC
    if _MODEL is None:
        _MODEL, _, _PREPROC = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=_DEVICE
        )
        _MODEL.eval()
    return _MODEL, _PREPROC

def embed_bgr_image(bgr: np.ndarray) -> np.ndarray:
    """
    Returns a 512-D float32 L2-normalized vector for a BGR crop.
    """
    model, preproc = _load_model()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    with torch.no_grad():
        img_t = preproc(pil).unsqueeze(0).to(_DEVICE)
        feats = model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    v = feats.cpu().numpy().astype("float32")[0]
    return v  # shape (512,)

def dim() -> int:
    model, _ = _load_model()
    # ViT-B/32 returns 512-D features
    return int(model.visual.output_dim)
