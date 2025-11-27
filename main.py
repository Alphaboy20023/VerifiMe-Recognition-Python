import cv2
import mediapipe as mp
import time
from collections import deque
from app.camera import Camera
from models.face import Face
from models.embedding import Embedding
from services.face_preprocessor import FacePreprocessor
from services.face_embedder import FaceEmbedder
from services.face_matcher import FaceMatcher
from data.user_db import load_registered_embeddings  # your existing user DB loader

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def smooth_box(prev_box, new_box, alpha=0.35):
    if prev_box is None:
        return new_box
    return tuple(int(prev_box[i] * (1 - alpha) + new_box[i] * alpha) for i in range(4))

def iou(boxA, boxB):
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

def match_faces(tracked, detected):
    matched = {}
    used = set()
    for tid, tbox in tracked.items():
        best_id = None
        best_score = 0
        for i, dbox in enumerate(detected):
            if i in used:
                continue
            score = iou(tbox, dbox)
            if score > best_score:
                best_score = score
                best_id = i
        if best_id is not None and best_score > 0.1:
            matched[tid] = detected[best_id]
            used.add(best_id)

    # Add new faces
    next_id = max(tracked.keys(), default=0)+1
    for i, dbox in enumerate(detected):
        if i not in used:
            matched[next_id] = dbox
            next_id += 1
    return matched

def main():
    cam = Camera()
    cam.open()

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    tracked_faces = {}
    prev_boxes = {}
    fps_hist = deque(maxlen=10)
    last = time.perf_counter()
    show_landmarks = False

    # Load registered embeddings from your DB
    registered_embeddings = load_registered_embeddings()  # returns list of Embedding objects
    matcher = FaceMatcher(threshold=0.5)
    embedder = FaceEmbedder()
    
    while True:
        frame = cam.get_frame()
        if frame is None:
            continue

        now = time.perf_counter()
        dt = now - last
        last = now
        fps = 1/dt if dt>0 else 0
        fps_hist.append(fps)
        fps_smoothed = sum(fps_hist)/len(fps_hist)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        detected_boxes = []

        if result.multi_face_landmarks:
            h, w = frame.shape[:2]
            for face_landmarks in result.multi_face_landmarks:
                xs = [pt.x for pt in face_landmarks.landmark]
                ys = [pt.y for pt in face_landmarks.landmark]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                x = int(min_x*w)
                y = int(min_y*h)
                bw = int((max_x-min_x)*w)
                bh = int((max_y-min_y)*h)
                detected_boxes.append((x, y, bw, bh))

        tracked_faces = match_faces(tracked_faces, detected_boxes)
        display = frame.copy()

        for fid, box in tracked_faces.items():
            smooth = smooth_box(prev_boxes.get(fid), box)
            prev_boxes[fid] = smooth
            x, y, bw, bh = smooth

            # Draw bounding box
            cv2.rectangle(display, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(display, f"ID {fid}", (x, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Generate embedding for this face
            embedding_vector = embedder.embed(frame, box)
            if embedding_vector is not None:
                best_match = matcher.find_best_match(embedding_vector, registered_embeddings)
                if best_match and best_match.is_match:
                    cv2.putText(display, f"User {best_match.face_id}", (x, y+bw+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.putText(display, f"FPS: {fps_smoothed:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("VerifiMe Face Recognition", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            show_landmarks = not show_landmarks

    cam.close()
    cv2.destroyAllWindows()
    face_mesh.close()

if __name__ == "__main__":
    main()
