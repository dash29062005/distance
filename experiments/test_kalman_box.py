import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel
from app.registration.motion_mask import MotionMask
from app.registration.mask_fusion import MaskFusion
from app.registration.contours import ContourExtractor
from app.registration.kalman_box import KalmanBox


def stack_row(imgs, max_h=350):
    """Resize each image to same height & stack horizontally."""
    processed = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = max_h / h
        resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        processed.append(resized)
    return np.hstack(processed)


def main():
    cam = WebcamCapture()
    cam.start()

    bg = BackgroundModel()
    flow = MotionMask()
    fusion = MaskFusion()
    contours = ContourExtractor()

    kf = KalmanBox()

    print("\nTest: Kalman Box Stabilizer (raw bbox vs smoothed bbox)\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ----- PIPELINE -----
        mask_bg, _ = bg.apply(frame)
        mask_flow = flow.update(frame)
        mask_fused = fusion.fuse(mask_bg, mask_flow)

        cnts, boxes = contours.extract(mask_fused, min_area=1500)

        vis = frame_bgr.copy()

        if boxes:
            # Use the largest detected box
            boxes = sorted(boxes, key=lambda b: b["area"], reverse=True)
            raw_box = boxes[0]["bbox"]
            x1, y1, x2, y2 = raw_box

            # Draw raw bbox (RED)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Smooth with Kalman
            smooth_box = kf.update(raw_box)
            sx1, sy1, sx2, sy2 = smooth_box

            # Draw smoothed bbox (GREEN)
            cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

            cv2.putText(vis, "RED = raw | GREEN = kalman-smoothed",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        else:
            # No contour → try prediction only
            pred = kf.predict_only()
            if pred is not None:
                px1, py1, px2, py2 = pred
                cv2.rectangle(vis, (px1, py1), (px2, py2), (255, 255, 0), 2)
                cv2.putText(vis, "Predict-only (no detection)",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 2)

        # ----- VISUALIZATION -----
        mask_display = cv2.cvtColor(mask_fused, cv2.COLOR_GRAY2BGR)
        row = stack_row([frame_bgr, mask_display, vis])

        # limit to screen width
        max_width = 1600
        if row.shape[1] > max_width:
            scale = max_width / row.shape[1]
            row = cv2.resize(row, (max_width, int(row.shape[0] * scale)))

        cv2.imshow("Kalman Box Test", row)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
