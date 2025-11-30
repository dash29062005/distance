import cv2
import time
import numpy as np


class CameraReadError(Exception):
    pass



class WebcamCapture:
    def __init__(self, device=0, width=1280, height=720, fps=30,
                 target_brightness=0.33):
        self.device = device
        self.width = width
        self.height = height
        self.fps_target = fps

        # Target brightness for stabilization
        self.target_brightness = target_brightness

        self.cap = None
        self.last_timestamp = None
        self.frame_id = 0

    # --------------------------------------------------------
    # START CAMERA
    # --------------------------------------------------------
    def start(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise CameraReadError("Could not open webcam.")

        # Natural auto-exposure (0.75 = Windows UVC auto)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)  # best-effort lower noise

        # Base resolution + FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps_target)

        self.last_timestamp = time.time()

    # --------------------------------------------------------
    # READ FRAME + LUMINANCE GAMMA STABILIZATION
    # --------------------------------------------------------
    def read(self, timeout=1.0):
        start_time = time.time()

        while True:
            success, frame = self.cap.read()
            if success:
                break
            if time.time() - start_time > timeout:
                raise CameraReadError("Timed out reading frame from camera.")

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize
        frame = cv2.resize(frame, (self.width, self.height))

        # FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_timestamp)
        self.last_timestamp = current_time

        # ----------------------------------------
        # LUMINANCE-ONLY GAMMA CORRECTION (HSV)
        # ----------------------------------------

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype("float32")

        # compute brightness from V channel
        raw_brightness = float(np.mean(hsv[:, :, 2])) / 255.0
        if raw_brightness < 0.01:
            raw_brightness = 0.01

        # gamma factor
        gamma = raw_brightness / self.target_brightness
        gamma = np.clip(gamma, 0.4, 2.2)  # safe range

        # apply gamma only to V (luminance)
        V = hsv[:, :, 2] / 255.0
        V_corrected = np.power(V, gamma)
        hsv[:, :, 2] = np.clip(V_corrected * 255.0, 0, 255)

        # back to RGB
        frame = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2RGB)

        # ----------------------------------------

        # Package
        self.frame_id += 1

        return {
            "frame": frame,
            "timestamp": current_time,
            "fps": fps,
            "frame_id": self.frame_id,
            "meta": {
                "brightness": raw_brightness,
                "gamma": gamma,
                "width": self.width,
                "height": self.height
            }
        }

    # --------------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------------
    def is_alive(self):
        return self.cap is not None and self.cap.isOpened()

    # --------------------------------------------------------
    # GET SETTINGS
    # --------------------------------------------------------
    def get_settings(self):
        return {
            "device": self.device,
            "width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "gain": self.cap.get(cv2.CAP_PROP_GAIN),
        }

    # --------------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------------
    def stop(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
