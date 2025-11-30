import cv2
from app.registration.embedder import EmbeddingGenerator

class ManualRegister:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0
        self.box_ready = False

        # Embedding generator (CLIP)
        self.embedder = EmbeddingGenerator()

    # Mouse callback
    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.x1, self.y1 = self.ix, self.iy
                self.x2, self.y2 = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.x1, self.y1 = self.ix, self.iy
            self.x2, self.y2 = x, y
            self.box_ready = True

    def select_roi(self, frame):
        """
        Shows frame, lets user draw ROI, returns:
          {
            'crop': numpy array (RGB),
            'bbox': (x1,y1,x2,y2),
            'embedding': 512-d vector
          }
        """
        clone = frame.copy()
        window_name = "Draw ROI (drag mouse, press ENTER)"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_event)

        while True:
            disp = clone.copy()

            if self.drawing or self.box_ready:
                cv2.rectangle(disp, (self.x1, self.y1), (self.x2, self.y2),
                              (0, 255, 0), 2)

            cv2.imshow(window_name, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            if key == 13 and self.box_ready:  # ENTER
                break

        cv2.destroyWindow(window_name)

        # Normalize coordinates
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])

        crop = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)

        # --------- CLIP embedding generation ----------
        emb = self.embedder.generate(crop)

        return {
            "crop": crop,
            "bbox": bbox,
            "embedding": emb
        }
