import cv2
import os
import time
import numpy as np
from deepface import DeepFace

# ----------------- Cosine Distance Function -----------------
def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ----------------- Load Known Faces -----------------
def load_known_faces(folder="known_faces"):
    face_db = {}
    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, filename)
            try:
                embedding = DeepFace.represent(img_path=path, model_name="Facenet512")[0]['embedding']
                face_db[filename] = embedding
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return face_db

# ----------------- Match Faces -----------------
def match_face(frame, known_faces, threshold=0.3):
    cv2.imwrite("temp.jpg", frame)
    try:
        live_embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet512")[0]['embedding']
        for name, stored_embedding in known_faces.items():
            distance = cosine_distance(live_embedding, stored_embedding)
            if distance < threshold:
                return f"✅ Match Found: {name}"
        return "❌ Face not recognized"
    except Exception as e:
        return f"⚠️ Error: {e}"

# ----------------- Main -----------------
def main():
    print("📦 Loading known faces...")
    known_faces = load_known_faces()
    print(f"✅ Loaded {len(known_faces)} known faces")

    cap = cv2.VideoCapture(0)
    print("📷 Webcam started. Auto-check every 3 seconds. Press 'Q' to quit.")

    last_check_time = 0
    check_interval = 3  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Face Recognition", frame)

        # Auto recognition
        if time.time() - last_check_time > check_interval:
            last_check_time = time.time()
            print("🔍 Matching face...")
            result = match_face(frame, known_faces)
            print(result)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
