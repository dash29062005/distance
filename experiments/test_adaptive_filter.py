import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel
from app.registration.motion_mask import MotionMask
from app.registration.mask_fusion import MaskFusion
from app.registration.adaptive_filter import AdaptiveAreaFilter


def draw_contours(base, contours, color):
    """Draw contours on copy of image."""
    canvas = np.zeros_like(base)
    cv2.drawContours(canvas, contours, -1, color, 2)
    return canvas


def resize_to_height(img, h=380):
    scale = h / img.shape[0]
    w = int(img.shape[1] * scale)
    return cv2.resize(img, (w, h))


def hstack_safe(imgs):
    imgs = [resize_to_height(im) for im in imgs]
    return np.hstack(imgs)


def vstack_safe(rows):
    max_w = max(row.shape[1] for row in rows)
    padded = []

    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack((row, pad))
        padded.append(row)

    return np.vstack(padded)


def main():
    cam = WebcamCapture()
    cam.start()

    bg = BackgroundModel()
    flow = MotionMask()
    fusion = MaskFusion()
    adaptive = AdaptiveAreaFilter()

    print("\nTest: AdaptiveAreaFilter (contours + filtered contours)\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ------- compute masks -------
        mask_bg, _ = bg.apply(frame)
        mask_flow = flow.update(frame)
        mask_fused = fusion.fuse(mask_bg, mask_flow)

        # ------- find contours -------
        binary = (mask_fused > 0).astype(np.uint8)
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # ------- adaptive filtering -------
        min_area = adaptive.update(binary, cnts)

        filtered = [c for c in cnts if cv2.contourArea(c) >= min_area]

        # ------- build visualization -------

        fused_vis = cv2.cvtColor(mask_fused, cv2.COLOR_GRAY2BGR)
        raw_contours_vis = draw_contours(fused_vis, cnts, (0, 0, 255))
        filtered_vis = draw_contours(fused_vis, filtered, (0, 255, 0))

        status = fused_vis.copy()
        cv2.putText(status, f"min_area={min_area}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # stack 2×2 view
        top = hstack_safe([fused_vis, raw_contours_vis])
        bottom = hstack_safe([filtered_vis, status])
        combined = vstack_safe([top, bottom])

        # cap width to ~1700px for your screen
        max_w = 1700
        if combined.shape[1] > max_w:
            scale = max_w / combined.shape[1]
            new_h = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (max_w, new_h))

        cv2.imshow("Adaptive Min-Area Test", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
