# Mediapipe/OpenCV logic
# face_detector.py
import cv2

class FaceDetector:
    """
    Detects faces in frames using OpenCV's Haar cascades.
    """

    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        """
        scaleFactor: How much the image size is reduced at each image scale
        minNeighbors: How many neighbors each rectangle should have to retain it
        minSize: Minimum possible object size
        """
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def detect_faces(self, frame):
        """
        Detect faces in a single frame.

        Returns a list of rectangles (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        return faces

    def draw_faces(self, frame, faces, color=(0, 255, 20), thickness=2):
        """
        Draw rectangles around detected faces.
        """
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame
