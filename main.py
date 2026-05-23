import cv2
import numpy as np
import os
import sqlite3
import pickle
from datetime import datetime, time

# ==========================================
# 1. CONFIGURATION
# ==========================================
video_url = "http://10.176.98.221:4747/video"  #Change to your DroidCam IP when ready
db_path = 'student_database.db'
encoding_file = 'sface_encodings.pickle' # The new pickle file!

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

SCHEDULE = {
    "P1": (time(9, 30), time(10, 30)),
    "P2": (time(10, 30), time(11, 30)),
    "P3": (time(11, 45), time(12, 45)),
    "P4": (time(13, 30), time(14, 30)),
    "P5": (time(14, 30), time(15, 30))
}

# ==========================================
# 2. DATABASE & LOGIC HELPERS
# ==========================================
def get_student_lookup():
    """Fetches a dictionary mapping Enrollment IDs to Names."""
    student_map = {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT enrollment_id, name FROM students")
        for r in cursor.fetchall():
            student_map[r[0]] = r[1]
    except Exception as e:
        print(f"⚠️ DB Warning: {e}")
    finally:
        conn.close()
    return student_map

def get_current_period():
    now_time = datetime.now().time()
    for period, (start, end) in SCHEDULE.items():
        if start <= now_time < end:
            return period
    return None

def mark_attendance(enroll_id):
    current_period = get_current_period()
    if not current_period:
        return "PAUSED"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S') 
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')

    # Check if already marked for THIS period today
    cursor.execute("SELECT timestamp FROM attendance WHERE enrollment_id = ? AND date = ?", (enroll_id, today_str))
    logs = cursor.fetchall()
    
    already_marked = False
    start_time, end_time = SCHEDULE[current_period]

    for log in logs:
        log_time = datetime.strptime(log[0], '%Y-%m-%d %H:%M:%S').time()
        if start_time <= log_time < end_time:
            already_marked = True
            break

    if not already_marked:
        cursor.execute("INSERT INTO attendance (enrollment_id, timestamp, date, time) VALUES (?, ?, ?, ?)", 
                       (enroll_id, dt_string, today_str, time_str))
        conn.commit()
        print(f"✅ LOGGED: {enroll_id} for {current_period}")
        status = "MARKED"
    else:
        status = "PRESENT" 

    conn.close()
    return status

# ==========================================
# 3. AI INITIALIZATION
# ==========================================
print("[INFO] Loading SFace Encodings and Models...")
if not os.path.exists(encoding_file):
    print(f"❌ Error: '{encoding_file}' not found. Run trainfaces.py first!")
    exit()

with open(encoding_file, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
class_ids = data["names"]
student_names = get_student_lookup()

# Initialize Models
detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

print(f"✅ Loaded {len(class_ids)} facial vectors. System ready!")

# ==========================================
# 4. MAIN VIDEO LOOP
# ==========================================
cap = cv2.VideoCapture(video_url)

while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to capture video.")
        break

    # Setup YuNet Input Size dynamically to match the camera frame
    h, w, _ = img.shape
    detector.setInputSize((w, h))

    # Detect Faces
    _, faces = detector.detect(img)
    
    # UI Status Bar
    curr_p = get_current_period()
    ui_status = f"Active: {curr_p}" if curr_p else "Status: BREAK/LUNCH"
    ui_color = (0, 255, 0) if curr_p else (0, 165, 255) 

    cv2.rectangle(img, (0,0), (640, 40), (30, 30, 30), cv2.FILLED)
    cv2.putText(img, ui_status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ui_color, 2)

    if faces is not None:
        for face in faces:
            # 1. Align and Extract Feature
            aligned_face = recognizer.alignCrop(img, face)
            face_feature = recognizer.feature(aligned_face)

            # 2. Compare against all known encodings
            identity = "Unknown"
            max_score = 0
            best_id_code = None

            for i, known_feat in enumerate(known_encodings):
                # Cosine Similarity check
                score = recognizer.match(face_feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                
                # The SFace threshold is typically 0.363. We use 0.40 to be slightly stricter.
                if score > 0.40 and score > max_score:
                    max_score = score
                    best_id_code = class_ids[i]

            # 3. Process the Match
            coords = face[:-1].astype(int)
            x, y, box_w, box_h = coords[0], coords[1], coords[2], coords[3]
            
            if best_id_code is not None:
                name = student_names.get(best_id_code, best_id_code)
                result = mark_attendance(best_id_code)
                
                box_color = (0, 255, 0) if result == "MARKED" else (255, 255, 0) # Green if newly marked, Cyan if already present
                if result == "PAUSED": box_color = (0, 165, 255) 
                
                # Draw Bounding Box & Text
                cv2.rectangle(img, (x, y), (x+box_w, y+box_h), box_color, 2)
                cv2.rectangle(img, (x, y-35), (x+box_w, y), box_color, cv2.FILLED)
                cv2.putText(img, name, (x+6, y-6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(img, f"Conf: {max_score:.2f}", (x, y+box_h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            else:
                # Unknown Face
                cv2.rectangle(img, (x, y), (x+box_w, y+box_h), (0, 0, 255), 2)
                cv2.rectangle(img, (x, y-35), (x+box_w, y), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Unknown", (x+6, y-6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('DAMS Live Feed - SFace Engine', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()