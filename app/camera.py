# camera reader
# camera.py
import cv2


class Camera:
    """
    Handles real-time video capture using the system's default camera.
    Designed to be used inside the VerifiMe SDK.
    """

    def __init__(self, camera_index=0):
        """
        camera_index: 0 = default webcam, 1/2 = external cameras
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_open = False

    def open(self):
        """Open the camera stream."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera.")
        self.is_open = True
        return True

    def close(self):
        """Release the camera stream."""
        if self.cap:
            self.cap.release() # release the camera so it frees up the device for other apps or future code runs
        self.is_open = False

    def get_frame(self):
        """
        Returns a single frame from the camera.
        Frame is returned as a numpy array (BGR).
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open. Call open() first.")

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")

        return frame

    def stream(self):
        """
        A generator that yields frames continuously.
        Use this for real-time recognition.
        """
        if not self.is_open:
            raise RuntimeError("Camera is not open. Call open() first.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def __del__(self):
        """Ensure camera is always released safely."""
        if self.is_open:
            self.close()
