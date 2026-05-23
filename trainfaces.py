import cv2
import os
import pickle
import numpy as np

# --- SETUP PATHS ---
BASE_PATH = "student_data"
DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"
ENCODING_FILE = "sface_encodings.pickle"

# --- VERIFY MODELS ---
if not os.path.exists(DETECTOR_MODEL) or not os.path.exists(RECOGNIZER_MODEL):
    print("❌ Error: ONNX model files not found in the 'models/' folder.")
    exit()

# --- INITIALIZE AI ---
# YuNet (Detector) and SFace (Recognizer)
detector = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

known_encodings = []
known_names = []

print("🚀 Starting High-Speed SFace Training...")

if not os.path.exists(BASE_PATH):
    print(f"❌ Error: '{BASE_PATH}' folder not found. Run register.py first.")
    exit()

# --- TRAINING LOOP ---
# Go through every student's folder in student_data
for student_id in os.listdir(BASE_PATH):
    student_dir = os.path.join(BASE_PATH, student_id)
    
    if os.path.isdir(student_dir):
        print(f"\nProcessing Student ID: {student_id}")
        
        # Go through the front, left, and right images
        for img_name in os.listdir(student_dir):
            if img_name.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(student_dir, img_name)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # 1. Tell YuNet the exact size of this image
                h, w, _ = img.shape
                detector.setInputSize((w, h))
                
                # 2. Detect the face
                _, faces = detector.detect(img)
                
                if faces is not None:
                    face = faces[0] # Grab the first face found
                    
                    # 3. Align the face (Straightens it based on landmarks)
                    aligned_face = recognizer.alignCrop(img, face)
                    
                    # 4. Extract the 128-dimensional feature vector
                    face_feature = recognizer.feature(aligned_face)
                    
                    # 5. Store it (Label it with the folder name, which is the ID)
                    known_encodings.append(face_feature)
                    known_names.append(student_id)
                    print(f"  ✅ Encoded: {img_name}")
                else:
                    print(f"  ⚠️ No face found in: {img_name}")

# --- SAVE THE BRAIN ---
with open(ENCODING_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"\n✨ Training Complete! Saved {len(known_names)} total facial vectors to '{ENCODING_FILE}'.")