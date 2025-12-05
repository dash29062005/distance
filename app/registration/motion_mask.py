import cv2
import numpy as np

class MotionMask:
    def __init__(self,
                 max_corners=200,
                 quality=0.01,
                 min_distance=7,
                 lk_win=21,
                 lk_levels=3,
                 mag_thresh=1.2,
                 min_points=30,
                 dilate_iter=3,
                 blur_ksize=7):
        
        # flow params
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.lk_win = (lk_win, lk_win)
        self.lk_levels = lk_levels
        self.mag_thresh = mag_thresh
        self.min_points = min_points

        # morphology
        self.dilate_iter = dilate_iter
        self.blur_ksize = blur_ksize

        # stored state
        self.prev_gray = None
        self.prev_pts = None

    # ------------------------------------------------
    def _init_points(self, gray):
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=self.quality,
            minDistance=self.min_distance
        )

    # ------------------------------------------------
    def update(self, frame):
        """Returns stable motion mask, or None if no motion."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # initialize points
        if self.prev_gray is None:
            self.prev_gray = gray
            self._init_points(gray)
            return np.zeros_like(gray)

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self._init_points(gray)

        # optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=self.lk_win,
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        good_prev = self.prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        # calc optical flow
        flow, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=self.lk_win,
            maxLevel=self.lk_levels,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # keep valid points only
        good_prev = self.prev_pts[status.flatten() == 1]
        good_next = flow[status.flatten() == 1]

        # if too few points → reset and return empty mask
        if len(good_prev) < 5:
            self.prev_gray = gray
            self.prev_pts = None
            return np.zeros_like(gray)

        # FIX: ensure flow shape is (N, 2)
        good_prev = good_prev.reshape(-1, 2)
        good_next = good_next.reshape(-1, 2)

        # motion vectors
        motion = good_next - good_prev

        # FIXED magnitude calculation
        mag = np.sqrt(motion[:, 0]**2 + motion[:, 1]**2)

        # threshold moving points
        moving = mag > self.mag_thresh

        if np.sum(moving) < self.min_points:
            # not enough motion → empty mask
            mask = np.zeros_like(gray)
        else:
            # sparse points → draw motion regions
            mask = np.zeros_like(gray)
            pts = good_prev[moving].astype(int)

            for (x, y) in pts:
                cv2.circle(mask, (x, y), 4, 255, -1)

            # blur & dilate to make solid regions
            mask = cv2.GaussianBlur(mask, (self.blur_ksize, self.blur_ksize), 0)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)

        # update state
        self.prev_gray = gray
        self.prev_pts = good_next.reshape(-1, 1, 2)

        return mask
