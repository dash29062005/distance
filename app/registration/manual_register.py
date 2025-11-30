import cv2

class ManualRegister:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.x1, self.y1 = 0, 0
        self.x2, self.y2 = 0, 0
        self.box_ready = False

    # ------------------------------
    # Mouse callback
    # ------------------------------
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

    # ------------------------------
    # ROI selection
    # ------------------------------
    def select_roi(self, frame):
        """
        Lets user draw ROI.
        Returns:
          crop (RGB numpy array) or None
          bbox (x1,y1,x2,y2) or None
        """
        clone = frame.copy()
        win = "Draw ROI (drag, ENTER)"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_event)

        while True:
            disp = clone.copy()

            # show live rectangle
            if self.drawing or self.box_ready:
                cv2.rectangle(disp, (self.x1, self.y1), (self.x2, self.y2),
                              (0, 255, 0), 2)

            cv2.imshow(win, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(1) & 0xFF

            # ENTER
            if key == 13 and self.box_ready:
                break

            # ESC → cancel
            if key == 27:
                cv2.destroyWindow(win)
                self._reset_state()
                return None, None

        cv2.destroyWindow(win)

        # Normalize coords
        x1, x2 = sorted([self.x1, self.x2])
        y1, y2 = sorted([self.y1, self.y2])

        # Zero-size box protection
        if x2 - x1 < 4 or y2 - y1 < 4:
            self._reset_state()
            return None, None

        crop = frame[y1:y2, x1:x2]
        bbox = (x1, y1, x2, y2)

        self._reset_state()
        return crop, bbox

    # ------------------------------
    # internal helper
    # ------------------------------
    def _reset_state(self):
        self.drawing = False
        self.box_ready = False
        self.ix = self.iy = -1
        self.x1 = self.y1 = 0
        self.x2 = self.y2 = 0
