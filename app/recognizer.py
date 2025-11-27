# matching logic
# app/recognizer.py
import cv2
import numpy as np
from collections import deque
from app.camera import Camera
from models.face import Face
from services.face_preprocessor import FacePreprocessor
from services.face_embedder import FaceEmbedder
from services.face_matcher import FaceMatcher
from data.user_db import load_registered_embeddings

class VerifiMeRecognizer:
    """
    Main recognizer class for the VerifiMe SDK.
    Handles camera capture, face detection, embedding, and matching.
    """

    def __init__(self, max_faces=5, matcher_threshold=0.5):
        self.camera = Camera()
        self.face_preprocessor = FacePreprocessor()
        self.embedder = FaceEmbedder()  # fallback to NumPy embeddings if TF Lite unavailable
        self.matcher = FaceMatcher(threshold=matcher_threshold)
        self.registered_embeddings = load_registered_embeddings()
        self.max_faces = max_faces

        self.tracked_faces = {}   # face_id -> bounding box
        self.prev_boxes = {}      # face_id -> smoothed box
        self.fps_history = deque(maxlen=10)
        self.show_landmarks = True
        self.running = False

    def smooth_box(self, prev_box, new_box, alpha=0.35):
        if prev_box is None:
            return new_box
        return tuple(int(prev_box[i] * (1 - alpha) + new_box[i] * alpha) for i in range(4))

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        inter = max(0, xB-xA) * max(0, yB-yA)
        if inter == 0:
            return 0
        areaA = boxA[2]*boxA[3]
        areaB = boxB[2]*boxB[3]
        return inter / (areaA + areaB - inter)

    def match_faces(self, tracked, detected):
        matched = {}
        used = set()
        for tid, tbox in tracked.items():
            best_id = None
            best_score = 0
            for i, dbox in enumerate(detected):
                if i in used:
                    continue
                score = self.iou(tbox, dbox)
                if score > best_score:
                    best_score = score
                    best_id = i
            if best_id is not None and best_score > 0.1:
                matched[tid] = detected[best_id]
                used.add(best_id)
        next_id = max(tracked.keys(), default=0)+1
        for i, dbox in enumerate(detected):
            if i not in used:
                matched[next_id] = dbox
                next_id += 1
        return matched

    def start(self):
        self.camera.open()
        self.running = True

    def stop(self):
        self.running = False
        self.camera.close()
        cv2.destroyAllWindows()

    def recognize_frame(self, frame):
        """
        Process a single frame: detect faces, compute embeddings, match against DB.
        Returns a list of dicts: [{face_id, bbox, embedding, match}]
        """
        faces_info = []
        detected_boxes = self.face_preprocessor.detect_faces(frame, max_faces=self.max_faces)
        self.tracked_faces = self.match_faces(self.tracked_faces, detected_boxes)

        for fid, box in self.tracked_faces.items():
            smooth = self.smooth_box(self.prev_boxes.get(fid), box)
            self.prev_boxes[fid] = smooth
            x, y, w, h = smooth

            embedding_vector = self.embedder.embed(frame, box)
            match = None
            if embedding_vector is not None:
                match = self.matcher.find_best_match(embedding_vector, self.registered_embeddings)

            faces_info.append({
                "face_id": fid,
                "bbox": smooth,
                "embedding": embedding_vector,
                "match": match
            })

        return faces_info

    def run_loop(self):
        """
        Starts a live camera loop. Use for testing/SDK demo.
        """
        self.start()
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils

        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=self.max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        last_time = cv2.getTickCount() / cv2.getTickFrequency()

        while self.running:
            frame = self.camera.get_frame()
            if frame is None:
                continue

            now = cv2.getTickCount() / cv2.getTickFrequency()
            dt = now - last_time
            last_time = now
            fps = 1/dt if dt > 0 else 0
            self.fps_history.append(fps)
            fps_smoothed = sum(self.fps_history)/len(self.fps_history)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            detected_boxes = []
            if result.multi_face_landmarks:
                h, w = frame.shape[:2]
                for face_landmarks in result.multi_face_landmarks:
                    xs = [pt.x for pt in face_landmarks.landmark]
                    ys = [pt.y for pt in face_landmarks.landmark]
                    x1, x2 = min(xs)*w, max(xs)*w
                    y1, y2 = min(ys)*h, max(ys)*h
                    detected_boxes.append((int(x1), int(y1), int(x2-x1), int(y2-y1)))

            self.tracked_faces = self.match_faces(self.tracked_faces, detected_boxes)
            display = frame.copy()

            for fid, box in self.tracked_faces.items():
                smooth = self.smooth_box(self.prev_boxes.get(fid), box)
                self.prev_boxes[fid] = smooth
                x, y, w, h = smooth
                embedding_vector = self.embedder.embed(frame, box)
                match = None
                if embedding_vector is not None:
                    match = self.matcher.find_best_match(embedding_vector, self.registered_embeddings)

                color = (0, 255, 0)
                label = f"ID {fid}"
                if match and match.is_match:
                    label += f" | User {match.face_id}"
                    color = (0, 255, 255)
                cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                cv2.putText(display, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.putText(display, f"FPS: {fps_smoothed:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("VerifiMe SDK Recognizer", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop()
                break
