import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel
from app.registration.motion_mask import MotionMask
from app.registration.mask_fusion import MaskFusion
from app.registration.auto_register import AutoRegister
from app.registration.confirm import HumanConfirm


def stack_row(imgs, max_h=350):
    out = []
    for img in imgs:
        h, w = img.shape[:2]
        scale = max_h / h
        img_r = cv2.resize(img, (int(w*scale), int(h*scale)))
        out.append(img_r)
    return np.hstack(out)


def main():
    cam = WebcamCapture()
    cam.start()

    bg = BackgroundModel()
    flow = MotionMask()
    fusion = MaskFusion()
    auto = AutoRegister(confirm=HumanConfirm()) # requires YES/NO popup

    print("\nTest: AutoRegister Pipeline")
    print("Press 'q' to quit.\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # ------------------- MASK PIPELINE -------------------
        mask_bg, _ = bg.apply(frame)
        mask_flow = flow.update(frame)
        mask_fused = fusion.fuse(mask_bg, mask_flow)
        mask_bgr = cv2.cvtColor(mask_fused, cv2.COLOR_GRAY2BGR)

        # ------------------- AUTO REGISTER -------------------
        # Fix argument order: mask_fused first, frame second
        reg = auto.update(mask_fused, frame)

        vis = frame_bgr.copy()

        if reg is not None:
            x1, y1, x2, y2 = reg["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"Registered ID: {reg['id']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

            print(f"[REGISTERED] ID={reg['id']} BBOX={reg['bbox']}")

        # ------------------- VISUALIZATION -------------------
        row = stack_row([frame_bgr, mask_bgr, vis])

        max_width = 1600
        if row.shape[1] > max_width:
            scale = max_width / row.shape[1]
            row = cv2.resize(row, (max_width, int(row.shape[0] * scale)))

        cv2.imshow("AutoRegister Test", row)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
