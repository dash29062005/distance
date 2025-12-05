import numpy as np
import cv2


class KalmanBox:
    """
    Smooth bounding boxes (cx, cy, w, h)
    with prediction when detection disappears.

    Now includes:
    - Hard reset (for NO confirmation)
    - Dead-frame counter (prevents collapse)
    - Min-size guard
    """

    def __init__(self, max_missing=5, min_size=10):
        self.kf = cv2.KalmanFilter(8, 4)

        # CONSTANT VELOCITY MODEL
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        self.kf.transitionMatrix[0, 4] = 1
        self.kf.transitionMatrix[1, 5] = 1
        self.kf.transitionMatrix[2, 6] = 1
        self.kf.transitionMatrix[3, 7] = 1

        # Measurement picks out cx,cy,w,h
        self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1
        self.kf.measurementMatrix[3, 3] = 1

        # Noise tuning
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)

        self.initialized = False
        self.missing_frames = 0
        self.max_missing = max_missing
        self.min_size = min_size

    # -------------------------------------------------------
    def _bbox_to_measure(self, bbox):
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, self.min_size)
        h = max(y2 - y1, self.min_size)
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w, h], dtype=np.float32)

    def _measure_to_bbox(self, m):
        cx, cy, w, h = m
        w = max(w, self.min_size)
        h = max(h, self.min_size)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return (x1, y1, x2, y2)

    # -------------------------------------------------------
    def reset(self):
        """FULL RESET for NO confirmation or new object."""
        self.initialized = False
        self.missing_frames = 0

    # -------------------------------------------------------
    def update(self, bbox):
        """
        Update Kalman with a new detection.
        Resets missing counter.
        """

        measurement = self._bbox_to_measure(bbox)

        # First frame → full init
        if not self.initialized:
            self.kf.statePost[:4, 0] = measurement.reshape(4)
            self.kf.statePost[4:, 0] = 0
            self.initialized = True
            self.missing_frames = 0
            return bbox

        # Normal predict → correct update
        _ = self.kf.predict()
        self.kf.correct(measurement.reshape(4, 1))

        self.missing_frames = 0

        smoothed = self.kf.statePost[:4, 0]
        return self._measure_to_bbox(smoothed)

    # -------------------------------------------------------
    def predict_only(self):
        """
        Use when contour disappeared temporarily.
        If object missing for too long → reset.
        """

        if not self.initialized:
            return None

        self.missing_frames += 1

        # too many missing frames → object gone
        if self.missing_frames > self.max_missing:
            self.reset()
            return None

        pred = self.kf.predict()
        smoothed = pred[:4, 0]
        return self._measure_to_bbox(smoothed)
