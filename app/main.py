import cv2
from app.camera.webcam import WebcamCapture
from app.registration.manual_register import ManualRegister
from app.registration.auto_register import AutoRegister

def draw_overlay(img, fps, brightness, gamma, auto_mode):
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(img, f"Brightness: {brightness*100:.1f}%", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(img, f"Gamma: {gamma:.2f}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(img, f"Auto-Reg: {auto_mode}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0,0,255) if auto_mode else (255,255,255), 2)


def main():
    cam = WebcamCapture()
    cam.start()

    manual = ManualRegister()
    auto = AutoRegister()

    auto_mode = False

    print("\nControls:")
    print("  r → manual register (draw ROI)")
    print("  a → toggle auto-registration")
    print("  q → quit\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        fps = pkg["fps"]
        brightness = pkg["meta"]["brightness"]
        gamma = pkg["meta"]["gamma"]

        # ---------- AUTO REGISTRATION ----------
        if auto_mode:
            new_objs = auto.update(frame)

            for obj in new_objs:
                print("\n[NEW OBJECT REGISTERED]")
                print("  ID:", obj["id"])
                print("  BBOX:", obj["bbox"])
                print("  Embedding shape:", obj["embedding"].shape)

                crop_bgr = cv2.cvtColor(obj["crop"], cv2.COLOR_RGB2BGR)
                cv2.imshow(f"Reg_ID_{obj['id']}", crop_bgr)

        # ---------- DRAW CAMERA OVERLAY ----------
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        draw_overlay(display, fps, brightness, gamma, auto_mode)
        cv2.imshow("Camera Preview", display)

        # ---------- INPUT ----------
        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            auto_mode = not auto_mode
            print("Auto-registration:", auto_mode)

        if key == ord('r'):
            print("Draw ROI...")
            obj = manual.select_roi(frame)   # <-- FIXED

            print("[Manual Registration]")
            print("  BBOX:", obj["bbox"])
            print("  Embedding shape:", obj["embedding"].shape)

            cv2.imshow(
                "Manual ROI",
                cv2.cvtColor(obj["crop"], cv2.COLOR_RGB2BGR)
            )

        if key == ord('q'):
            break

    cam.stop()


if __name__ == "__main__":
    main()
