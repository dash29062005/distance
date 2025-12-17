import cv2
from app.camera.webcam import WebcamCapture
from app.registration.background_model import BackgroundModel


def main():
    cam = WebcamCapture(width=640, height=360, fps=20)
    cam.start()

    bg = BackgroundModel()

    while True:
        pkg = cam.read()
        frame = pkg["frame"]  # RGB

        mask, stats = bg.apply(frame)

        # Display original + mask side by side
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mask_col = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combo = cv2.hconcat([disp, mask_col])

        cv2.putText(combo,
                    f"warmup={stats['warmup']}  fg_ratio={stats['fg_ratio']:.3f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0), 2)

        cv2.imshow("BackgroundModel Test (left=frame, right=mask)", combo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
