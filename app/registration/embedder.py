import torch
import numpy as np
import cv2
import os
import json
from PIL import Image   # ← REQUIRED for CLIP preprocess
import time

import clip


class EmbeddingGenerator:
    """
    Lightweight CLIP embedding generator.
    Converts RGB crop → normalized 1024D vector.
    """

    def __init__(self, device=None, model_name="RN50"):
        # DEVICE SELECT
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"[Embedder] Loading CLIP ({model_name}) on {self.device} ...")

        # LOAD MODEL
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

        # STORAGE FOLDER
        self.store_dir = os.path.join("data", "embeddings")
        os.makedirs(self.store_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    def _prepare(self, rgb):
        """Convert RGB np array → CLIP tensor"""
        if rgb is None or rgb.size == 0:
            return None

        # Resize to CLIP input size
        rgb_resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        rgb_resized = rgb_resized.astype(np.uint8)

        # Convert to CLIP preprocess pipeline
        pil_img = Image.fromarray(rgb_resized)
        tensor = self.preprocess(pil_img).unsqueeze(0)

        return tensor.to(self.device)

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def get_embedding(self, rgb):
        """Returns: (1024,) float32 embedding or None"""
        x = self._prepare(rgb)
        if x is None:
            return None

        features = self.model.encode_image(x).float()

        # L2 normalize
        features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy().astype(np.float32)

    # ----------------------------------------------------------------------
    def compare(self, emb1, emb2):
        """Cosine similarity in [0,1]"""
        if emb1 is None or emb2 is None:
            return 0.0

        emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)

        return float(np.dot(emb1, emb2))

    # ----------------------------------------------------------------------
    def save_embedding(self, obj_id, bbox, embedding):
        """Save embedding into data/embeddings/<id>.json"""

        path = os.path.join(self.store_dir, f"{obj_id}.json")

        data = {
            "id": int(obj_id),
            "bbox": [float(v) for v in bbox],
            "embedding": embedding.tolist(),
            "timestamp": time.time(),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[Embedder] Saved → {path}")

    # ----------------------------------------------------------------------
    def load_embedding(self, obj_id):
        """Load embedding JSON"""
        path = os.path.join(self.store_dir, f"{obj_id}.json")
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            data = json.load(f)

        data["embedding"] = np.array(data["embedding"], dtype=np.float32)
        return data
