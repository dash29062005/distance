import numpy as np
import cv2
import time


class AdaptiveAreaFilter:
    """
    Adjusts min_area threshold dynamically based on:
        - mask density
        - contour statistics
        - scene noise level
        - recent stability

    API:
        filt = AdaptiveAreaFilter()
        min_area = filt.update(mask_fused, contours)
    """

    def __init__(self,
                 base_min_area=1200,
                 max_min_area=6000,
                 smooth_alpha=0.25,
                 noise_floor=0.012,
                 growth_factor=1.6,
                 shrink_factor=0.7):
        
        self.base_min_area = base_min_area
        self.max_min_area = max_min_area
        self.smooth_alpha = smooth_alpha

        self.noise_floor = noise_floor
        self.growth_factor = growth_factor
        self.shrink_factor = shrink_factor

        # state
        self.current_min_area = base_min_area
        self.prev_fg_ratio = 0.0
        self.prev_area = base_min_area
        self.last_update = time.time()


    def _compute_fg_ratio(self, mask):
        """Foreground density (fraction of pixels = 1)."""
        nonzero = np.count_nonzero(mask)
        return nonzero / (mask.size + 1e-6)


    def update(self, mask, contours):
        """
        Inputs:
            mask (0/255)
            contours from this mask

        Output:
            new_min_area (int)
        """

        # Normalize binary mask
        binary = (mask > 0).astype(np.uint8)

        fg_ratio = self._compute_fg_ratio(binary)

        # Compute typical contour area
        areas = [cv2.contourArea(c) for c in contours]
        median_area = np.median(areas) if len(areas) > 0 else 0

        # --- Decision Logic --------------------------------------

        new_area = self.current_min_area

        # 1. If scene very noisy (too many FG pixels) → increase area threshold
        if fg_ratio > self.noise_floor:
            new_area = min(
                int(new_area * self.growth_factor),
                self.max_min_area
            )

        # 2. If contours are consistently small → reduce threshold
        elif median_area > 0 and median_area < new_area * 0.6:
            new_area = max(
                int(new_area * self.shrink_factor),
                self.base_min_area
            )

        # 3. If stable scene → slowly return to baseline
        elif fg_ratio < self.noise_floor * 0.5:
            new_area = int(new_area * 0.9)
            new_area = max(new_area, self.base_min_area)

        # Smooth gradual transitions
        new_area = int(
            self.smooth_alpha * new_area +
            (1 - self.smooth_alpha) * self.current_min_area
        )

        # Apply
        self.current_min_area = new_area
        self.prev_fg_ratio = fg_ratio
        self.last_update = time.time()

        return self.current_min_area
