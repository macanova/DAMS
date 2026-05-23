# DAMS - Deep Attendance Monitoring System

A smart AI-powered attendance and security monitoring system using Face Recognition and Computer Vision.

## 🚀 Features

- 🎭 Face Detection & Recognition
- 🧑‍🎓 Student Registration System
- 📸 Multiple Face Angle Dataset Collection
- ✅ Automatic Attendance Marking
- 🚨 Intruder Detection System
- 🗂 Attendance Report Viewer
- 🧠 Face Encoding Training
- 💾 SQLite Database Integration
- 📊 Attendance Management
- 🔐 Security Monitoring

---

## 🛠 Technologies Used

- Python
- OpenCV
- NumPy
- SQLite3
- ONNX Face Models
- Deep Learning Based Face Recognition

---

## 📁 Project Structure

```bash
DAMS/
│
├── main.py
├── register.py
├── trainfaces.py
├── viewattendance.py
├── viewreports.py
├── setupdatabase.py
├── testmodels.py
│
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
│
├── student_data/
├── intruders/
├── student_database.db
└── sface_encodings.pickle
