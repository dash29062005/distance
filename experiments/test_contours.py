import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel
from app.registration.motion_mask import MotionMask
from app.registration.mask_fusion import MaskFusion
from app.registration.contours import ContourExtractor


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

    print("\nTest: Contours (frame | fused_mask | contours)")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ---------- MASK PIPELINE ----------
        mask_bg, _ = bg.apply(frame)
        mask_flow = flow.update(frame)
        mask_fused = fusion.fuse(mask_bg, mask_flow)

        # ---------- GET CONTOURS ----------
        cnts, boxes = contours.extract(mask_fused, min_area=1500)

        # Draw contours on a copy
        vis = frame_bgr.copy()
        for box in boxes:
            x1, y1, x2, y2 = box["bbox"]
            cx, cy = box["centroid"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # ---------- PRINT STATS ----------
        if boxes:
            print("\nContours Detected:", len(boxes))
            for b in boxes:
                print(f"Area: {b['area']:.1f}, BBOX: {b['bbox']}, Centroid: {b['centroid']}")

        # ---------- VISUALIZATION ----------
        m_fused = cv2.cvtColor(mask_fused, cv2.COLOR_GRAY2BGR)
        row = stack_row([frame_bgr, m_fused, vis])

        # Limit max width
        max_width = 1600
        if row.shape[1] > max_width:
            scale = max_width / row.shape[1]
            row = cv2.resize(row, (max_width, int(row.shape[0] * scale)))

        cv2.imshow("Contours Test", row)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
