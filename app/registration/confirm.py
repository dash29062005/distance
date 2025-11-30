import cv2

class ConfirmDialog:
    def __init__(self, window_name="Confirm Registration"):
        self.window_name = window_name

    def ask(self, crop_rgb):
        """
        Displays a blocking window with the crop.
        User presses 'y' to accept, 'n' to reject.
        Returns: True or False
        """
        img_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, img_bgr)

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key == ord('y') or key == ord('Y'):
                cv2.destroyWindow(self.window_name)
                return True

            if key == ord('n') or key == ord('N'):
                cv2.destroyWindow(self.window_name)
                return False

            # allow escape as reject
            if key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                return False
