import cv2
import numpy as np
import pickle
import os

# 1. Setup Paths
detector_model = "models/face_detection_yunet_2023mar.onnx"
recognizer_model = "models/face_recognition_sface_2021dec.onnx"
encoding_file = "sface_encodings.pickle"

# 2. Check if files exist
if not os.path.exists(detector_model) or not os.path.exists(recognizer_model):
    print("❌ Error: Model files not found in 'models/' folder!")
    exit()

# 3. Initialize Models
detector = cv2.FaceDetectorYN.create(detector_model, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(recognizer_model, "")

# 4. Load Encodings (if they exist)
known_data = None
if os.path.exists(encoding_file):
    with open(encoding_file, "rb") as f:
        known_data = pickle.load(f)
    print(f"✅ Loaded {len(known_data['names'])} student encodings.")

# 5. Start Camera Test
cap = cv2.VideoCapture(0) # Change to DroidCam URL if needed
print("📸 Camera starting... Look at the lens! Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Set Input Size for YuNet
    h, w, _ = frame.shape
    detector.setInputSize((w, h))

    # Detection
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Draw Bounding Box (first 4 values in face array)
            coords = face[:-1].astype(np.int32)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)

            # Recognition Logic
            if known_data:
                # Align and get feature for the current face
                face_aligned = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_aligned)

                # Compare against known students using Cosine Similarity
                for i, known_feat in enumerate(known_data["encodings"]):
                    score = recognizer.match(face_feature, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                    
                    if score > 0.363: # Threshold for SFace
                        name = known_data["names"][i]
                        cv2.putText(frame, f"{name} ({score:.2f})", (coords[0], coords[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        break
                else:
                    cv2.putText(frame, "Unknown", (coords[0], coords[1]-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Diagnostic Mode - YuNet + SFace", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()