# 

import cv2
import os
from deepface import DeepFace

# Load known face embeddings
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

# Match webcam face to known faces
def match_face(frame, known_faces, threshold=0.3):
    cv2.imwrite("temp.jpg", frame)
    try:
        live_embedding = DeepFace.represent(img_path="temp.jpg", model_name="Facenet512")[0]['embedding']
        for name, stored_embedding in known_faces.items():
            distance = DeepFace.dst.findCosineDistance(live_embedding, stored_embedding)
            if distance < threshold:
                return f"✅ Match Found: {name}"
        return "❌ Face not recognized"
    except Exception as e:
        return f"⚠️ Error: {e}"

# Main webcam loop
def main():
    print("📦 Loading known faces...")
    known_faces = load_known_faces()
    print(f"✅ Loaded {len(known_faces)} known faces")

    cap = cv2.VideoCapture(0)
    print("📷 Webcam started. Press 'C' to check face. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Face Recognition", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            print("🔍 Matching face...")
            result = match_face(frame, known_faces)
            print(result)

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
