import cv2
import os
import math
import sqlite3
import pytesseract
import time
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp

# Optional: set Tesseract path (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Load known face embeddings ---
def load_face_db(folder="known_faces"):
    db = {}
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            path = os.path.join(folder, file)
            try:
                emb = DeepFace.represent(img_path=path, model_name="Facenet512")[0]["embedding"]
                db[file] = emb
            except:
                continue
    return db

# --- Match face ---
def match_face(frame, db, threshold=0.3):
    cv2.imwrite("live_face.jpg", frame)
    try:
        live_emb = DeepFace.represent(img_path="live_face.jpg", model_name="Facenet512")[0]["embedding"]
        for name, ref_emb in db.items():
            dist = DeepFace.dst.findCosineDistance(live_emb, ref_emb)
            if dist < threshold:
                return name
    except:
        return None
    return None

# --- OCR for ID card ---
def extract_id_info(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    name, usn, cls = "", "", ""
    for line in text.split("\n"):
        if "Name" in line:
            name = line.split(":")[-1].strip()
        elif "USN" in line:
            usn = line.split(":")[-1].strip()
        elif "Class" in line:
            cls = line.split(":")[-1].strip()
    return name, usn, cls

# --- Blink Detection ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_blink_ratio(landmarks, eye):
    top = landmarks[eye[1]]
    bottom = landmarks[eye[4]]
    left = landmarks[eye[0]]
    right = landmarks[eye[3]]

    vertical = math.dist([top.x, top.y], [bottom.x, bottom.y])
    horizontal = math.dist([left.x, left.y], [right.x, right.y])
    return vertical / horizontal

def detect_blink(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            left = get_blink_ratio(landmarks, LEFT_EYE)
            right = get_blink_ratio(landmarks, RIGHT_EYE)
            avg_ratio = (left + right) / 2
            if avg_ratio < 0.28:  # easier blink detection
                return True
    return False

# --- Save attendance to DB ---
def mark_attendance(name, usn):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("attendance.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        usn TEXT,
        timestamp TEXT)''')
    c.execute("INSERT INTO attendance (name, usn, timestamp) VALUES (?, ?, ?)", (name, usn, now))
    conn.commit()
    conn.close()

# --- Main Function ---
def main():
    print("📦 Loading known faces...")
    db = load_face_db()
    print(f"✅ Loaded {len(db)} known faces.\nStarting camera...")

    cap = cv2.VideoCapture(0)
    last_trigger_time = time.time()
    blink_last_status = None
    trigger_printed = False  # new flag

    # Mediapipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = face_detection.process(rgb)

        status_text = ""
        status_color = (0, 255, 0)

        # Draw box if face detected
        if detections.detections:
            for detection in detections.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h),
                        int(bboxC.width * w), int(bboxC.height * h))
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        # Trigger recognition every 3 sec if face present
        if time.time() - last_trigger_time > 3 and detections.detections:
            if not trigger_printed:
                print("\n🔄 Auto Recognition Triggered...")
                trigger_printed = True  # prevent spam printing

            if detect_blink(frame):
                if blink_last_status != "blink":
                    print("✅ Liveness confirmed (blink)")
                    blink_last_status = "blink"

                name_file = match_face(frame, db)
                if name_file:
                    print(f"😊 Face matched: {name_file}")
                    name_id, usn, _ = extract_id_info(frame)
                    mark_attendance(name_id or name_file, usn or "UNKNOWN")
                    print(f"📌 Attendance marked for {name_id or name_file} ({usn or 'UNKNOWN'})")
                    status_text = f"ATTENDANCE MARKED: {name_id or name_file}"
                    status_color = (0, 255, 0)
                else:
                    print("❌ Face not recognized")
                    status_text = "FACE NOT RECOGNIZED"
                    status_color = (0, 0, 255)
            else:
                if blink_last_status != "noblink":
                    print("❌ No blink detected. Skipping...")
                    blink_last_status = "noblink"
                status_text = "NO BLINK DETECTED"
                status_color = (0, 255, 255)

            last_trigger_time = time.time()
            trigger_printed = False  # reset for next cycle

        elif not detections.detections:
            if blink_last_status != "noface":
                print("🚫 No face detected in frame")
                blink_last_status = "noface"
            status_text = "NO FACE DETECTED"
            status_color = (0, 0, 255)

        # Overlay status text on feed
        if status_text:
            cv2.putText(frame, status_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Smart Attendance (Auto Mode)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
