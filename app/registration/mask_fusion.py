import cv2
import numpy as np

class MaskFusion:
    def __init__(self,
                 bg_weight=0.6,
                 flow_weight=0.4,
                 blur_ksize=7,
                 dilate_iter=2,
                 erode_iter=1,
                 min_fill_area=1500,
                 temporal_alpha=0.4):
        
        self.bg_weight = bg_weight
        self.flow_weight = flow_weight
        self.blur_ksize = blur_ksize
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.min_fill_area = min_fill_area
        
        # temporal smoothing buffer
        self.prev_mask = None
        self.temporal_alpha = temporal_alpha


    def _solidify(self, mask):
        """Fill holes + morphological closing."""
        kernel = np.ones((5, 5), np.uint8)

        # close gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # dilate slightly
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)

        # optional slight erosion to refine boundaries
        mask = cv2.erode(mask, kernel, iterations=self.erode_iter)

        # smooth edges
        mask = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)

        return mask


    def _fill_holes(self, mask):
        """Fill internal holes (binary)."""
        h, w = mask.shape
        flood = mask.copy()

        mask_filled = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)

        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)

        mask_filled = cv2.bitwise_or(mask, flood_inv)

        return mask_filled


    def _temporal_smooth(self, mask):
        """Exponential moving average to stabilize."""
        if self.prev_mask is None:
            self.prev_mask = mask.astype(np.float32)
            return mask

        self.prev_mask = (
            self.temporal_alpha * mask.astype(np.float32)
            + (1 - self.temporal_alpha) * self.prev_mask
        )

        return self.prev_mask.astype(np.uint8)


    def fuse(self, mask_bg, mask_flow):
        """
        Combine background mask + motion mask → stable solid object mask.
        """
        # Normalize inputs (ensure uint8 0/255)
        mask_bg = (mask_bg > 0).astype(np.uint8) * 255
        mask_flow = (mask_flow > 0).astype(np.uint8) * 255

        # Weighted sum fusion
        fused = (
            self.bg_weight * mask_bg.astype(np.float32) +
            self.flow_weight * mask_flow.astype(np.float32)
        )

        fused = np.clip(fused, 0, 255).astype(np.uint8)

        # Threshold to binary
        _, fused = cv2.threshold(fused, 80, 255, cv2.THRESH_BINARY)

        # Solidify regions
        fused = self._solidify(fused)

        # Fill internal gaps
        fused = self._fill_holes(fused)

        # Remove tiny blobs
        cnts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(fused)

        for c in cnts:
            if cv2.contourArea(c) >= self.min_fill_area:
                cv2.drawContours(cleaned, [c], -1, 255, -1)

        # Temporal smoothing
        cleaned = self._temporal_smooth(cleaned)

        return cleaned
