class Embedding:
    def __init__(self, vector, face_id=None, source="live"):
        self.vector = vector        # numpy array or list
        self.face_id = face_id      # match to a Face object
        self.source = source        # "live", "database", etc.
