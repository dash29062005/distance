import cv2
import numpy as np

from app.registration.motion_mask import MotionMask
from app.camera.webcam import WebcamCapture


def stack_h(img1, img2):
    """Stack two images horizontally with auto-resize."""
    h = min(img1.shape[0], img2.shape[0])
    img1 = cv2.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
    img2 = cv2.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))
    return np.hstack((img1, img2))


def main():
    cam = WebcamCapture()
    cam.start()

    motion = MotionMask()

    print("\nTest: MotionMask (frame | mask)\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]

        mask = motion.update(frame)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        view = stack_h(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), mask_bgr)
        # resize view to max width 1600px (fits your 1700px screen)
        max_w = 1500
        if view.shape[1] > max_w:
            scale = max_w / view.shape[1]
            new_h = int(view.shape[0] * scale)
            view = cv2.resize(view, (max_w, new_h))

        cv2.putText(view, "frame | motion mask", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("MotionMask Test", view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
