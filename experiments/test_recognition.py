import cv2
import numpy as np

from app.camera.webcam import WebcamCapture
from app.registration.embedder import EmbeddingGenerator
from app.registration.recognition import Recognizer
from app.registration.vector_store import VectorStore   # Qdrant wrapper


def stack_h(a, b, scale=0.7):
    """Simple horizontal stack of frame + crop."""
    h = int(a.shape[0] * scale)
    a = cv2.resize(a, (int(a.shape[1] * scale), h))
    b = cv2.resize(b, (int(b.shape[1] * scale), h))
    return np.hstack([a, b])


def main():
    cam = WebcamCapture()
    cam.start()

    embedder = EmbeddingGenerator()
    store = VectorStore(dim=1024)
    recognizer = Recognizer(store, embedder)

    crop = None

    print("\nTest: Recognition system")
    print("Press SPACE → capture crop & identify")
    print("Press q → quit\n")

    while True:
        pkg = cam.read()
        frame = pkg["frame"]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display = frame_bgr.copy()
        cv2.putText(display, "Press SPACE to identify crop",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        if crop is not None:
            view = stack_h(display, crop)
        else:
            view = display

        cv2.imshow("Recognition Test", view)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            # take center crop 120x120 for quick test
            h, w = frame.shape[:2]
            c = 120
            x1 = w//2 - c//2
            y1 = h//2 - c//2
            x2 = x1 + c
            y2 = y1 + c

            crop = frame[y1:y2, x1:x2]

            # identify
            result = recognizer.identify(crop, (x1, y1, x2, y2))

            if result["is_new"]:
                status = f"NEW ID {result['id']} | score={result['score']:.3f}"
            else:
                status = f"MATCH ID {result['id']} | score={result['score']:.3f}"

            print(status)

            # draw crop location
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if key == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
