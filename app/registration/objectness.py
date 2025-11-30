import cv2
import numpy as np


class ObjectnessFilter:
    def __init__(self, use_cuda=True):
        self.use_cuda = use_cuda and cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.use_cuda:
            print("[Objectness] Using CUDA Sobel + GPU acceleration.")
        else:
            print("[Objectness] CUDA unavailable → using CPU fallback.")

    def _compute_sobel_gpu(self, roi):
        gpu_roi = cv2.cuda_GpuMat()
        gpu_roi.upload(roi)

        sobel_x = cv2.cuda.Sobel(gpu_roi, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.cuda.Sobel(gpu_roi, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.cuda.magnitude(sobel_x, sobel_y).download()

        return mag

    def _compute_sobel_cpu(self, roi):
        gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag

    def compute(self, frame_rgb, bbox, mode="score"):
        """
        frame_rgb: full frame (RGB)
        bbox: (x1, y1, x2, y2)
        mode: "score" or "full"
        """

        x1, y1, x2, y2 = bbox
        roi = frame_rgb[y1:y2, x1:x2]

        if roi.size == 0:
            return 0.0 if mode == "score" else {
                "edge_density": 0,
                "contrast": 0,
                "gradient_strength": 0,
                "score": 0
            }

        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # ---- Sobel Gradient (GPU or CPU) ----
        if self.use_cuda:
            mag = self._compute_sobel_gpu(gray)
        else:
            mag = self._compute_sobel_cpu(gray)

        # ---- Feature 1: Edge Density ----
        edges = mag > 25
        edge_density = np.sum(edges) / edges.size

        # ---- Feature 2: Gradient Strength ----
        gradient_strength = float(np.mean(mag)) / 255.0

        # ---- Feature 3: Local Contrast ----
        contrast = float(np.std(gray)) / 255.0

        # ---- Combined Objectness Score ----
        score = (
            0.45 * edge_density +
            0.35 * gradient_strength +
            0.20 * contrast
        )

        # Clamp score 0–1
        score = max(0.0, min(1.0, score))

        if mode == "score":
            return float(score)

        # Full breakdown for logs/visualization
        return {
            "edge_density": float(edge_density),
            "gradient_strength": float(gradient_strength),
            "contrast": float(contrast),
            "score": float(score)
        }
