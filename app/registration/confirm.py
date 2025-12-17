import cv2
import numpy as np
import time


class HumanConfirm:
    """
    Simple YES/NO popup for confirming auto-registration.
    Blocks until user presses:
        y → YES
        n → NO
        q → cancel (treated as NO)
    """

    def __init__(self, window="Confirm Object"):
        self.window = window

    def ask(self, crop_rgb):
        """
        Show crop and wait for user input.

        Returns:
            True  → YES
            False → NO
        """
        img = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
        disp = img.copy()

        cv2.putText(disp, "Confirm this object? (y/n)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        cv2.imshow(self.window, disp)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('y'):
                cv2.destroyWindow(self.window)
                return True

            if key == ord('n') or key == ord('q'):
                cv2.destroyWindow(self.window)
                return False
