import cv2
import numpy as np


class ContourExtractor:
    """
    Extract clean object contours from a fused binary mask.
    
    API:
        cnts, boxes = contour.extract(mask, min_area=1200)

    Outputs:
        cnts  -> list of contours
        boxes -> list of dicts:
                 {
                    'bbox': (x1, y1, x2, y2),
                    'area': float,
                    'centroid': (cx, cy)
                 }
    """

    def __init__(self,
                 morph_kernel=5,
                 closing_iter=2,
                 dilation_iter=1):
        
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )
        self.closing_iter = closing_iter
        self.dilation_iter = dilation_iter

    # -----------------------------------------------------------
    def _preprocess(self, mask):
        """Fill holes + close gaps to form solid contours."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                self.kernel, iterations=self.closing_iter)
        mask = cv2.dilate(mask, self.kernel, iterations=self.dilation_iter)
        return mask

    # -----------------------------------------------------------
    def extract(self, mask, min_area=1000):
        """
        Extract contours and bounding boxes.

        mask → binary mask (0/255)
        min_area → remove tiny blobs
        """

        if mask is None or mask.size == 0:
            return [], []

        # ensure binary
        mask = (mask > 0).astype(np.uint8) * 255

        # preprocess for better contours
        mask_clean = self._preprocess(mask)

        # find contours
        contours, _ = cv2.findContours(
            mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return [], []

        final_cnts = []
        final_boxes = []

        for c in contours:
            area = cv2.contourArea(c)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            cx = x + w / 2
            cy = y + h / 2

            final_cnts.append(c)
            final_boxes.append({
                "bbox": (x, y, x + w, y + h),
                "area": float(area),
                "centroid": (float(cx), float(cy))
            })

        return final_cnts, final_boxes
