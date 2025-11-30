import numpy as np
import cv2


class KalmanBox:
    def __init__(self):
        """
        Kalman filter with 6 state variables:
        x, y, vx, vy, w, h
        """
        self.kf = cv2.KalmanFilter(6, 4)  # (state_dim=6, measurement_dim=4)

        # State transition matrix
        dt = 1.0  # assuming ~30 FPS, we update each frame

        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, 0, 0],
            [0, 1, 0, dt, 0, 0],
            [0, 0, 1, 0,  0, 0],
            [0, 0, 0, 1,  0, 0],
            [0, 0, 0, 0,  1, 0],
            [0, 0, 0, 0,  0, 1],
        ], dtype=np.float32)

        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0],  # y
            [0, 0, 0, 0, 1, 0],  # w
            [0, 0, 0, 0, 0, 1],  # h
        ], dtype=np.float32)

        # Process noise (higher = more adaptable)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03

        # Measurement noise (lower = more trust in detection)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.01

        # Initial error covariance
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.initialized = False
        self.stable_score = 0.0

    def _bbox_to_state(self, bbox):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h

    def correct(self, bbox):
        """
        bbox: (x1, y1, x2, y2)
        Update Kalman filter with new measurement.
        """
        cx, cy, w, h = self._bbox_to_state(bbox)

        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)

        if not self.initialized:
            # Initialize state
            self.kf.statePost = np.array([[cx], [cy], [0], [0], [w], [h]], dtype=np.float32)
            self.initialized = True
            return bbox

        self.kf.correct(meas)

        # Stability scoring
        pred = self.kf.predict()
        px, py, vx, vy, pw, ph = pred.flatten()

        deviation = np.sqrt((cx - px)**2 + (cy - py)**2)  # pixel drift
        size_change = abs(w - pw) + abs(h - ph)

        # High stability → high score
        self.stable_score = np.exp(-(deviation + 0.5 * size_change) / 40)

        return (int(px - pw/2), int(py - ph/2), int(px + pw/2), int(py + ph/2))

    def predict(self):
        """
        Predict next position without measurement.
        """
        if not self.initialized:
            return None, 0.0

        pred = self.kf.predict()
        px, py, vx, vy, pw, ph = pred.flatten()
        bbox = (int(px - pw/2), int(py - ph/2), int(px + pw/2), int(py + ph/2))

        return bbox, self.stable_score
