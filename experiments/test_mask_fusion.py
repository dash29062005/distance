import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel
from app.registration.motion_mask import MotionMask
from app.registration.mask_fusion import MaskFusion


def resize_to_height(img, h=350):
    """Resize image to fixed height, keep aspect ratio."""
    ih, iw = img.shape[:2]
    scale = h / ih
    new_w = int(iw * scale)
    return cv2.resize(img, (new_w, h))


def stack_2x2(a, b, c, d, h=350):
    """Create a clean 2×2 grid."""
    a = resize_to_height(a, h)
    b = resize_to_height(b, h)
    c = resize_to_height(c, h)
    d = resize_to_height(d, h)

    top = np.hstack((a, b))
    bottom = np.hstack((c, d))
    return np.vstack((top, bottom))


def main():
    cam = WebcamCapture()
    cam.start()

    bg = BackgroundModel()
    flow = MotionMask()
    fusion = MaskFusion()

    print("\nTest: MaskFusion (2×2 view)\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # masks
        mask_bg, _ = bg.apply(frame)
        mask_flow = flow.update(frame)
        mask_fused = fusion.fuse(mask_bg, mask_flow)

        # convert masks to BGR for consistent display
        m1 = cv2.cvtColor(mask_bg, cv2.COLOR_GRAY2BGR)
        m2 = cv2.cvtColor(mask_flow, cv2.COLOR_GRAY2BGR)
        m3 = cv2.cvtColor(mask_fused, cv2.COLOR_GRAY2BGR)

        # 2×2 panel layout
        panel = stack_2x2(frame_bgr, m1, m2, m3)

        # Add labels
        cv2.putText(panel, "Frame", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0,255,0), 2)
        cv2.putText(panel, "BG Mask", (panel.shape[1]//2 + 20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(panel, "Flow Mask", (20, panel.shape[0]//2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.putText(panel, "Fused Mask",
                    (panel.shape[1]//2 + 20, panel.shape[0]//2 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Mask Fusion 2x2", panel)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
