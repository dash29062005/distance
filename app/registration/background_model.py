import cv2
import numpy as np
import time


class BackgroundModel:
    """
    Robust background subtractor for auto-registration pipeline.
    Uses MOG2 + post-processing + warmup logic.

    API:
        bg = BackgroundModel()
        mask, stats = bg.apply(frame_rgb)

    Output:
        mask  -> 0/255 uint8 binary mask
        stats -> dict { 'fg_ratio', 'mean', 'nonzero', 'timestamp' }
    """

    def __init__(
        self,
        history=200,
        var_threshold=16,
        detect_shadows=True,
        kernel_size=3,
        warmup_frames=40
    ):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                (kernel_size, kernel_size))

        self.frame_count = 0
        self.warmup_frames = warmup_frames
        self.last_mask = None

    # ----------------------------------------------------------------------

    def apply(self, frame_rgb):
        """
        Apply background subtractor.
        Input:  RGB frame (HxWx3, uint8)
        Output: mask (0/255), stats
        """

        self.frame_count += 1

        # BGR is better for MOG2 internally
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Raw mask from MOG2 (0, 127 for shadows, 255 for fg)
        raw_mask = self.bg.apply(frame_bgr)

        # Ignore mask during warmup (stabilize background model)
        if self.frame_count < self.warmup_frames:
            h, w = raw_mask.shape
            empty_mask = np.zeros((h, w), dtype=np.uint8)
            return empty_mask, {
                "fg_ratio": 0.0,
                "mean": 0.0,
                "nonzero": 0,
                "timestamp": time.time(),
                "warmup": True
            }

        # Threshold shadow regions to 0, keep only strong fg
        _, mask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # Optional: small dilate to make contours slightly bigger
        mask = cv2.dilate(mask, self.kernel, iterations=1)

        # Compute stats
        nonzero = int(np.count_nonzero(mask))
        fg_ratio = nonzero / (mask.size + 1e-6)
        stats = {
            "fg_ratio": fg_ratio,
            "mean": float(np.mean(mask) / 255.0),
            "nonzero": nonzero,
            "timestamp": time.time(),
            "warmup": False
        }

        self.last_mask = mask
        return mask, stats

    # ----------------------------------------------------------------------

    def get_state(self):
        return {
            "frame_count": self.frame_count,
            "warmup_left": max(0, self.warmup_frames - self.frame_count)
        }
