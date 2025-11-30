import cv2
import numpy as np
from collections import deque

from .objectness import ObjectnessFilter
from .kalman_box import KalmanBox
from .confirm import ConfirmDialog
from .embedder import EmbeddingGenerator


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class AutoRegister:
    def __init__(self,
                 min_area=1200,
                 confirm_frames=8,
                 iou_threshold=0.4,
                 bg_learn_frames=40):

        # Background learning
        self.bg_frames = deque(maxlen=bg_learn_frames)
        self.static_bg = None
        self.bg_learned = False

        # Motion background subtractor
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=30, detectShadows=True)

        # Filters
        self.min_area = min_area
        self.confirm_frames = confirm_frames
        self.iou_threshold = iou_threshold

        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

        # Submodules
        self.obj_filter = ObjectnessFilter()
        self.confirm = ConfirmDialog()
        self.embedder = EmbeddingGenerator()

        # Internal state
        self.candidates = []
        self.registered = {}
        self.next_id = 1
        self.frame_idx = 0

    # -------------------------------------------------------
    # Background median learner
    # -------------------------------------------------------
    def _update_static_background(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.bg_frames.append(gray)

        if len(self.bg_frames) == self.bg_frames.maxlen:
            self.static_bg = np.median(
                np.stack(list(self.bg_frames)), axis=0).astype("uint8")
            self.bg_learned = True
            print("[AutoRegister] Static background learned.")

    def _difference_from_background(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.absdiff(gray, self.static_bg)

    # -------------------------------------------------------
    # Mask cleaning
    # -------------------------------------------------------
    def _clean_mask(self, mask):
        _, th = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, self.kernel)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, self.kernel)
        return th

    def _extract_boxes(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)

            if w*h < self.min_area:
                continue

            if w/h < 0.25 or h/w < 0.25:
                continue

            boxes.append((x, y, x+w, y+h))

        return boxes

    # -------------------------------------------------------
    # Main update loop
    # -------------------------------------------------------
    def update(self, frame_rgb):
        self.frame_idx += 1

        # -------------------------
        # Background warm-up
        # -------------------------
        if not self.bg_learned:
            self._update_static_background(frame_rgb)
            return []

        # -------------------------
        # Motion detection
        # -------------------------
        fg = self.bg.apply(frame_rgb)
        mask = self._clean_mask(fg)
        boxes = self._extract_boxes(mask)

        verified_boxes = []

        # -------------------------
        # Background difference check
        # -------------------------
        diff_map = self._difference_from_background(frame_rgb)

        for box in boxes:
            x1,y1,x2,y2 = box
            diff_crop = diff_map[y1:y2, x1:x2]
            if np.mean(diff_crop) < 20:
                continue  # too similar → background
            verified_boxes.append(box)

        # -------------------------
        # Objectness + Kalman candidates
        # -------------------------
        new_regs = []
        updated_candidates = []

        for box in verified_boxes:
            obj_score = self.obj_filter.compute(
                frame_rgb, box, mode="score")

            if obj_score < 0.25:
                continue  # weak objectness

            # MATCH WITH EXISTING CANDIDATES
            matched = False
            for c in self.candidates:
                if iou(box, c["bbox"]) > self.iou_threshold:
                    c["bbox"] = box
                    c["frames"] += 1
                    c["kalman"].correct(box)
                    c["last_frame"] = self.frame_idx
                    matched = True

                    # Check Kalman stability
                    if (c["frames"] >= self.confirm_frames and
                        c["kalman"].stable_score > 0.5 and
                        not c["confirmed"]):

                        # SHOW CROP → HUMAN YES/NO
                        x1,y1,x2,y2 = box
                        crop = frame_rgb[y1:y2, x1:x2]

                        accept = self.confirm.ask(crop)
                        if not accept:
                            c["confirmed"] = True
                            continue

                        # Generate embedding
                        emb = self.embedder.generate(crop)

                        obj_id = self.next_id
                        self.next_id += 1

                        self.registered[obj_id] = {
                            "id": obj_id,
                            "bbox": box,
                            "crop": crop,
                            "embedding": emb
                        }

                        new_regs.append(self.registered[obj_id])
                        c["confirmed"] = True

                    break

            if not matched:
                # New candidate
                updated_candidates.append({
                    "bbox": box,
                    "frames": 1,
                    "kalman": KalmanBox(),
                    "confirmed": False,
                    "last_frame": self.frame_idx
                })

        # Update candidate pool
        self.candidates.extend(updated_candidates)
        self.candidates = [
            c for c in self.candidates
            if self.frame_idx - c["last_frame"] < 25
        ]

        return new_regs
