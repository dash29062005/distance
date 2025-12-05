import cv2
from app.camera.webcam import WebcamCapture
from app.registration.confirm import HumanConfirm


def main():
    cam = WebcamCapture()
    cam.start()

    confirm = HumanConfirm()

    print("\nPress 'c' to confirm a crop | 'q' to quit\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("Confirm Test", frame_bgr)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Take center crop just for testing
            h, w = frame.shape[:2]
            cx1 = w // 4
            cy1 = h // 4
            cx2 = 3 * w // 4
            cy2 = 3 * h // 4

            crop = frame[cy1:cy2, cx1:cx2]

            result = confirm.ask(crop)
            print("User confirmed:", result)

        if key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
