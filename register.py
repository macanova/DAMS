import cv2
import os
import sqlite3

# --- SETUP ---
BASE_PATH = "student_data"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
if not os.path.exists(BASE_PATH): os.makedirs(BASE_PATH)

# Initialize YuNet
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320))

def is_clear(image, threshold=60):
    """Checks if the image is sharp."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold

def get_face_angle(face):
    """Calculates head pose for front and 90-degree profiles."""
    re_x, le_x, nose_x = face[4], face[6], face[8]
    
    dist_right = abs(nose_x - re_x)
    dist_left = abs(nose_x - le_x)
    
    # If one eye is hidden (distance is 0 or extremely small), it's a 90-degree profile
    if dist_right <= 5: 
        return "right"
    if dist_left <= 5: 
        return "left"
        
    ratio = dist_left / dist_right

    # Broadened the ratio to accept a standard front face
    if 0.5 <= ratio <= 2.0:
        return "front"
    elif ratio > 2.0:
        return "right" 
    elif ratio < 0.5:
        return "left"  
    else:
        return "transitioning"

def register_simple_ai():
    print("🎓 DAMS V2.0 - CLEAN REGISTRATION")
    enroll_id = input("1. Enrollment ID: ").strip().upper()
    name = input("2. Full Name: ").strip()
    
    student_dir = os.path.join(BASE_PATH, enroll_id)
    if not os.path.exists(student_dir): os.makedirs(student_dir)

    video_url = "http://10.176.98.221:4747/video"  # Change to your DroidCam IP when ready
    cap = cv2.VideoCapture(video_url)
    angles = ["front", "left", "right"]
    
    for target_angle in angles:
        print(f"\n📸 TARGET: {target_angle.upper()}")
        print("Wait for 'PERFECT', then press 's' to Capture, or 'q' to Quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            display_frame = frame.copy()
            h, w, _ = frame.shape
            detector.setInputSize((w, h))
            
            _, faces = detector.detect(frame)
            status_text = f"Looking for {target_angle.upper()}..."
            status_color = (0, 0, 255) 

            if faces is not None:
                face = faces[0] 
                coords = face[:-1].astype(int)
                
                face_crop = frame[max(0, coords[1]):coords[1]+coords[3], max(0, coords[0]):coords[0]+coords[2]]
                
                if face_crop.size > 0:
                    clear = is_clear(face_crop)
                    current_angle = get_face_angle(face)
                    
                    if not clear:
                        status_text = "BLURRY! Hold still."
                    elif current_angle != target_angle:
                        status_text = f"Looking {current_angle}. Turn {target_angle.upper()}."
                    else:
                        status_text = f"PERFECT! Press 's' to Save."
                        status_color = (0, 255, 0) 
                
                # Simple clean bounding box
                cv2.rectangle(display_frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), status_color, 2)
                
            cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.imshow("Registration", display_frame)
            
            key = cv2.waitKey(1)
            # Only allow save if the AI confirms the angle and clarity (Green status)
            if key == ord('s') and status_color == (0, 255, 0): 
                img_name = f"{enroll_id}_{target_angle}.jpg"
                cv2.imwrite(os.path.join(student_dir, img_name), frame)
                print(f"✨ SUCCESS: {target_angle.upper()} captured!")
                break
                
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Registration Cancelled.")
                return

    # Database Entry
    conn = sqlite3.connect('student_database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO students (enrollment_id, name) VALUES (?, ?)", (enroll_id, name))
    conn.commit()
    conn.close()
    
    print(f"\n✅ Registration Complete! Images stored in {student_dir}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_simple_ai()