import numpy as np
import cv2
import json
import os


class MonoDistance:
    """
    Mono camera distance estimator using:
      - bounding box height
      - focal length from calibration
      - real world object height
    """

    def __init__(self, calib_path="data/calibration/charuco_calibration.json"):
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"Calibration file missing: {calib_path}")

        with open(calib_path, "r") as f:
            calib = json.load(f)

        self.camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
        self.distortion = np.array(calib["distortion_coefficients"], dtype=np.float32)
        self.focal_length = float(self.camera_matrix[1, 1])  # fy

    # --------------------------------------------------------------
    def undistort(self, frame):
        """Apply undistortion once at capture layer."""
        return cv2.undistort(frame, self.camera_matrix, self.distortion)

    # --------------------------------------------------------------
    def estimate(self, bbox, real_height):
        """
        bbox → (x1, y1, x2, y2)
        real_height → meters

        RETURNS distance in meters
        """
        x1, y1, x2, y2 = bbox
        pixel_height = abs(y2 - y1)

        if pixel_height <= 0:
            return None

        # pinhole camera model
        distance = (real_height * self.focal_length) / pixel_height
        return float(distance)
