import numpy as np
from scipy.optimize import linear_sum_assignment
from app.registration.kalman_box import KalmanBox


class Track:
    """Single DeepSORT-style track."""
    def __init__(self, track_id, bbox, embedding):
        self.id = track_id
        self.kf = KalmanBox()
        self.bbox = self.kf.update(bbox)
        self.embedding = embedding
        self.hits = 1
        self.missed = 0


class DeepSortTracker:
    """
    Simplified DeepSORT pipeline for your system.
    Input each frame:
        detections = [
            { "bbox": (x1,y1,x2,y2), "embedding": emb, "id": maybe_id_from_recognizer }
        ]

    Output:
        list of tracks → [ {id, bbox} ]
    """

    def __init__(self,
                 max_age=15,
                 iou_threshold=0.3,
                 embed_weight=0.6,
                 iou_weight=0.4):
        
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.embed_weight = embed_weight
        self.iou_weight = iou_weight

        self.next_id = 1
        self.tracks = []


    # ------------------------- METRICS -------------------------
    def _iou(self, boxA, boxB):
        x1 = max(boxA[0], boxB[0])
        y1 = max(boxA[1], boxB[1])
        x2 = min(boxA[2], boxB[2])
        y2 = min(boxA[3], boxB[3])

        interW = max(0, x2 - x1)
        interH = max(0, y2 - y1)
        inter = interW * interH

        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        union = areaA + areaB - inter

        if union == 0:
            return 0.0
        return inter / union


    def _cosine(self, a, b):
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(np.dot(a, b))


    # --------------------- ASSIGNMENT --------------------------
    def _build_cost_matrix(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return None

        T = len(self.tracks)
        D = len(detections)

        cost = np.zeros((T, D))

        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):

                iou = self._iou(track.bbox, det["bbox"])
                emb_sim = self._cosine(track.embedding, det["embedding"])

                # Higher similarity → LOWER cost
                score = (self.iou_weight * iou) + (self.embed_weight * emb_sim)
                cost[i, j] = 1 - score

        return cost


    # --------------------- MAIN UPDATE -------------------------
    def update(self, detections):
        """
        detections = list of { bbox, embedding }
        Returns active tracks: [ {id, bbox} ]
        """

        # Predict all tracks
        predictions = []
        for t in self.tracks:
            pred = t.kf.predict_only()
            if pred is not None:
                t.bbox = pred
            predictions.append(t)

        # If no detections → age tracks
        if len(detections) == 0:
            for t in self.tracks:
                t.missed += 1
            self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
            return [ {"id": t.id, "bbox": t.bbox} for t in self.tracks ]

        # Build cost matrix
        cost = self._build_cost_matrix(detections)

        if cost is None:
            assigned_rows = []
            assigned_cols = []
        else:
            row_ind, col_ind = linear_sum_assignment(cost)
            assigned_rows = row_ind
            assigned_cols = col_ind

        # Keep track of which detections got used
        used_dets = set()

        # Update matched tracks
        for r, c in zip(assigned_rows, assigned_cols):
            if cost is not None and cost[r, c] > 0.7:
                # too dissimilar → treat as unmatched
                continue

            track = self.tracks[r]
            det = detections[c]

            track.bbox = track.kf.update(det["bbox"])
            track.embedding = det["embedding"]
            track.hits += 1
            track.missed = 0

            used_dets.add(c)

        # Add new tracks for unmatched detections
        for idx, det in enumerate(detections):
            if idx not in used_dets:
                t = Track(self.next_id, det["bbox"], det["embedding"])
                self.next_id += 1
                self.tracks.append(t)

        # Age old tracks
        alive_tracks = []
        for t in self.tracks:
            if t.missed > self.max_age:
                continue
            alive_tracks.append(t)

        self.tracks = alive_tracks

        return [ {"id": t.id, "bbox": t.bbox} for t in self.tracks ]
