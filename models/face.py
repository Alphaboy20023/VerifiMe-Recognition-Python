class Face:
    def __init__(self, box, score, landmarks=None, face_id=None):
        self.box = box              # (x, y, w, h)
        self.score = score          # detection confidence
        self.landmarks = landmarks  # mediapipe landmarks
        self.face_id = face_id      # tracking ID
        self.crop = None            # cropped face image (filled later)

    def set_crop(self, crop):
        self.crop = crop
