class MatchResult:
    def __init__(self, distance, threshold, is_match, face_id=None):
        self.distance = distance
        self.threshold = threshold
        self.is_match = is_match
        self.face_id = face_id

    def __str__(self):
        return (
            f"MatchResult(distance={self.distance:.4f}, "
            f"threshold={self.threshold}, match={self.is_match})"
        )
