import cv2
from app.camera import WebcamCapture

def main():
    cam = WebcamCapture(width=1280, height=720, fps=30)
    cam.start()

    print("[Camera Test] Started. Press Q to quit.")

    while True:
        pkg = cam.read()

        frame = pkg["frame"]              # RGB
        fps = pkg["fps"]
        brightness = pkg["meta"]["brightness"]
        gamma = pkg["meta"]["gamma"]

        # show frame
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # overlay
        cv2.putText(disp, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(disp, f"Brightness: {brightness:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(disp, f"Gamma: {gamma:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Camera Test", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()

if __name__ == "__main__":
    main()
