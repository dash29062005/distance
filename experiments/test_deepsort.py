import cv2
import numpy as np
from app.camera.webcam import WebcamCapture
from app.tracking.tracker_deepsort import DeepSortTracker
from app.registration.kalman_box import KalmanBox


def draw_bbox(img, bbox, track_id, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, f"ID {track_id}", (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def fake_embedding():
    """Random embedding to simulate recognizer output."""
    v = np.random.randn(1024).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def main():
    print("\nRunning DeepSORT tracker test...\n")

    cam = WebcamCapture()
    cam.start()

    tracker = DeepSortTracker()

    print("Press Q to quit.\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w = display.shape[:2]

        # -------------------------------------------------------------
        # SIMULATED DETECTION:
        # Here we create a fake moving box that moves like a tracked object.
        # Replace this with REAL detections after integration.
        # -------------------------------------------------------------

        cx = int(w/2 + 80*np.sin(pkg["frame_id"] / 20))
        cy = int(h/2 + 50*np.cos(pkg["frame_id"] / 25))

        box = (cx-60, cy-40, cx+60, cy+40)
        emb = fake_embedding()

        detections = [
            {"bbox": box, "embedding": emb}
        ]

        # UPDATE TRACKER
        tracks = tracker.update(detections)

        # DRAW TRACKS
        for t in tracks:
            draw_bbox(display, t["bbox"], t["id"])

        cv2.imshow("DeepSORT Test", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
