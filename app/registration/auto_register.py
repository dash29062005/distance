import cv2
import numpy as np
import time

from app.registration.kalman_box import KalmanBox


class AutoRegister:
    """
    Auto-registration pipeline using fused mask → contour → Kalman box.
    
    Behavior:
      - Detects **one candidate object at a time**
      - Smooths box with KalmanBox
      - Sends crop to confirm module (YES/NO)
      - YES → return registered crop
      - NO → reset and wait for next candidate
    """

    def __init__(
        self,
        min_area=2000,
        pad=12,
        confirm=None
    ):
        self.min_area = min_area
        self.pad = pad
        self.confirm = confirm     # confirm.ask(crop) → True/False

        self.kalman = KalmanBox()
        self.current_bbox = None
        self.state = "search"      # "search", "tracking", "confirm"

        self.last_crop = None
        self.frame_counter = 0

    # -------------------------------------------------------------
    def _pad_box(self, box, w, h):
        x1, y1, x2, y2 = box
        x1 -= self.pad
        y1 -= self.pad
        x2 += self.pad
        y2 += self.pad
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        return (x1, y1, x2, y2)

    # -------------------------------------------------------------
    def _extract_biggest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0

        for c in cnts:
            a = cv2.contourArea(c)
            if a > best_area:
                best_area = a
                best = c

        if best is None or best_area < self.min_area:
            return None

        x, y, w, h = cv2.boundingRect(best)
        return (x, y, x + w, y + h)

    # -------------------------------------------------------------
    def update(self, mask_fused, frame_rgb):
        """
        INPUT:
            mask_fused → stable foreground mask
            frame_rgb → actual frame to crop from

        OUTPUT:
            None OR dict { "crop", "bbox" }
        """

        h, w = mask_fused.shape
        self.frame_counter += 1

        # 1) SEARCH MODE → find a new contour
        if self.state == "search":
            bbox = self._extract_biggest_contour(mask_fused)
            if bbox is None:
                return None

            bbox = self._pad_box(bbox, w, h)
            self.current_bbox = bbox
            self.kalman.reset()
            self.state = "tracking"
            return None

        # 2) TRACKING MODE → update Kalman
        if self.state == "tracking":
            bbox = self._extract_biggest_contour(mask_fused)

            if bbox is not None:
                bbox = self._pad_box(bbox, w, h)
                smooth = self.kalman.update(bbox)
            else:
                # missing frame → use prediction
                smooth = self.kalman.predict_only()
                if smooth is None:
                    self.state = "search"
                    self.current_bbox = None
                    return None

            self.current_bbox = smooth

            # Prepare crop and ask for YES/NO
            x1, y1, x2, y2 = smooth
            crop = frame_rgb[y1:y2, x1:x2]
            self.last_crop = crop

            # Switch to confirm mode (one-shot evaluate)
            self.state = "confirm"
            return None

        # 3) CONFIRM MODE → ask human
        if self.state == "confirm":
            if self.last_crop is None:
                self.state = "search"
                return None

            answer = self.confirm.ask(self.last_crop)

            if answer is True:
                # Return registered object
                result = {
                    "bbox": self.current_bbox,
                    "crop": self.last_crop,
                    "id": self.frame_counter  # Use frame_counter as a unique ID
                }

                # reset for next object
                self.state = "search"
                self.current_bbox = None
                self.kalman.reset()
                self.last_crop = None
                return result

            else:
                # NO → discard object, restart search
                self.state = "search"
                self.current_bbox = None
                self.kalman.reset()
                self.last_crop = None
                return None
