import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.manual_register import ManualRegister
from app.registration.embedder import EmbeddingGenerator


def main():
    cam = WebcamCapture()
    cam.start()

    manual = ManualRegister()
    embedder = EmbeddingGenerator()

    print("\nTest: Embedding Generator")
    print("Press 'r' → Draw ROI and compute embedding")
    print("Press 'q' → Quit\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]

        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(display, "Press 'r' to crop + embed", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Embedding Test", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            roi, coords = manual.select_roi(frame)

            if roi is None or roi.size == 0:
                print("[WARN] Empty ROI")
                continue

            emb = embedder.get_embedding(roi)

            print("\n[Embedding computed]")
            print("  ROI:", coords)
            print("  Shape:", emb.shape)
            print("  First 5 values:", emb[:5])

            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            cv2.imshow("ROI", roi_bgr)

        if key == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
