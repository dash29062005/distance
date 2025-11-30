import torch
import clip
from PIL import Image
import numpy as np


class EmbeddingGenerator:
    def __init__(self, device="cuda"):
        """
        Loads CLIP ViT-B/32 on GPU.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"[Embedder] Loading CLIP on {self.device}...")

        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def generate(self, image_rgb):
        """
        image_rgb: numpy array (H, W, 3) in RGB
        Returns L2-normalized 512-d embedding vector.
        """
        pil_img = Image.fromarray(image_rgb)

        # Preprocess → CLIP format
        img_proc = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(img_proc)
            emb = emb / emb.norm(dim=-1, keepdim=True)  # Normalize

        vec = emb.squeeze().cpu().numpy().astype(np.float32)
        return vec
