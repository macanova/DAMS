import streamlit as st
import sqlite3
import pandas as pd
import cv2
import os
import time
import pickle
import numpy as np
import shutil
import io
from fpdf import FPDF
import streamlit.components.v1 as components
from datetime import datetime, time as dt_time

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
DB_FILE = 'student_database.db'
BASE_PATH = 'student_data' 
INTRUDER_DIR = 'intruders'  
ENCODING_FILE = 'sface_encodings.pickle'
DEFAULT_DROIDCAM = "http://10.176.98.200:4747/video" 

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNIZER_MODEL = "models/face_recognition_sface_2021dec.onnx"

# NEW: Camera Location for Intruder Logs
CAMERA_LOCATION = "Main-Entrance" # Change this based on where the camera is deployed!

# Custom Schedule
PERIODS = ["P1", "P2", "P3", "P4", "P5"]
SCHEDULE_MAP = {
    "P1": (dt_time(9, 30), dt_time(10, 30)),
    "P2": (dt_time(10, 30), dt_time(11, 30)),
    "P3": (dt_time(11, 45), dt_time(12, 45)),
    "P4": (dt_time(13, 30), dt_time(14, 30)),
    "P5": (dt_time(14, 30), dt_time(15, 30))
}

# Ensure required directories exist
for d in [BASE_PATH, INTRUDER_DIR, "models"]:
    if not os.path.exists(d): os.makedirs(d)

st.set_page_config(page_title="DAMS By ARGUS", page_icon="🎓", layout="wide")

# ==========================================
# 2. AUTOFILL BLOCKER & CSS STYLING
# ==========================================
components.html(
    """
    <script>
    document.addEventListener('DOMContentLoaded', function() {
        const observer = new MutationObserver(() => {
            const inputs = window.parent.document.querySelectorAll('input[type="text"]');
            inputs.forEach(input => {
                input.setAttribute('autocomplete', 'off');
                if (!input.hasAttribute('data-randomized')) {
                    input.setAttribute('name', 'field_' + Math.random().toString(36).substring(7));
                    input.setAttribute('data-randomized', 'true');
                }
            });
        });
        observer.observe(window.parent.document.body, { childList: true, subtree: true });
    });
    </script>
    """,
    height=0
)

st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. DATABASE INITIALIZER
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS students (
            enrollment_id TEXT PRIMARY KEY, admission_number TEXT, name TEXT, department TEXT, course TEXT, year TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
            enrollment_id TEXT, timestamp TEXT, date TEXT, time TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT)''')
    
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", ("admin", "admin123"))
    except sqlite3.IntegrityError:
        pass 
        
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 4. AI MODELS & HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_ai_models():
    d = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320, 320))
    r = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")
    return d, r

detector, recognizer = load_ai_models()

def retrain_sface_brain():
    known_encodings, known_names = [], []
    for student_id in os.listdir(BASE_PATH):
        student_dir = os.path.join(BASE_PATH, student_id)
        if os.path.isdir(student_dir):
            for img_name in os.listdir(student_dir):
                if img_name.endswith((".jpg", ".png")):
                    img = cv2.imread(os.path.join(student_dir, img_name))
                    if img is None: continue
                    detector.setInputSize((img.shape[1], img.shape[0]))
                    _, faces = detector.detect(img)
                    if faces is not None:
                        feat = recognizer.feature(recognizer.alignCrop(img, faces[0]))
                        known_encodings.append(feat)
                        known_names.append(student_id)
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

def get_connection(): return sqlite3.connect(DB_FILE)

def login_user(username, password):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    data = c.fetchall()
    conn.close()
    return data

def run_query(query, params=()):
    conn = get_connection()
    try: df = pd.read_sql_query(query, conn, params=params)
    except Exception: return pd.DataFrame()
    finally: conn.close()
    return df

def map_timestamp_to_period(ts_str):
    try:
        t = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S').time()
        for p_name, (start, end) in SCHEDULE_MAP.items():
            if start <= t < end: return p_name
    except: pass
    return None
def generate_pdf_report(df, report_label):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"DAMS Attendance - {report_label}", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 8) # Shrunk font slightly to fit 8 columns
    # Widths for: ID, Name, Dept, Days, Periods, Hours, Day_%, Period_%
    col_widths = [20, 35, 15, 20, 25, 20, 25, 25] 
    
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, str(col).replace("_", " "), border=1, align="C")
    pdf.ln()
    
    pdf.set_font("Arial", "", 8)
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            pdf.cell(col_widths[i], 10, str(row[col]), border=1, align="C")
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

def generate_period_pdf_report(df, date_str):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, f"DAMS Daily Period Matrix - {date_str}", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 10)
    col_widths = [30, 60, 20, 20, 20, 20, 20] 
    for i, col in enumerate(df.columns):
        pdf.cell(col_widths[i], 10, str(col), border=1, align="C")
    pdf.ln()
    
    pdf.set_font("Arial", "", 10)
    for _, row in df.iterrows():
        for i, col in enumerate(df.columns):
            pdf.cell(col_widths[i], 10, str(row[col]), border=1, align="C")
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 5. MAIN APP INTERFACE & ROUTING
# ==========================================

if 'logged_in' not in st.session_state: 
    st.session_state['logged_in'] = False

# --- VIEW 1: LOGIN SCREEN ---
if not st.session_state['logged_in']:
    st.title("🔐 DAMS Secure Login")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login_user(username, password):
                st.session_state['logged_in'] = True
                st.rerun()
            else:
                st.error("Invalid Credentials")

# --- VIEW 2: MAIN DASHBOARD ---
else:
    st.sidebar.title("🎓 DAMS Super-UI")
    st.sidebar.success("✅ Admin Online")
    
    if st.sidebar.button("🔓 Logout"):
        st.session_state['logged_in'] = False
        st.session_state['camera_running'] = False 
        st.rerun()
        
    st.sidebar.markdown("---")
    page = st.sidebar.radio("Navigate", [
        "🏠 Dashboard", 
        "🔴 Live AI Monitor", 
        "🚨 Intruder Logs", 
        "📊 Period Matrix", 
        "📅 Monthly Reports", 
        "📝 Register Student", 
        "📂 Database"
    ])
    st.sidebar.markdown("---")

    # ==========================================
    # PAGE 1: DASHBOARD
    # ==========================================
    if page == "🏠 Dashboard":
        st.title("📈 System Dashboard")
        students_df = run_query("SELECT enrollment_id, name, department, year FROM students")
        today_str = datetime.now().strftime('%Y-%m-%d')
        attendance_df = run_query("SELECT DISTINCT enrollment_id FROM attendance WHERE date=?", (today_str,))
        
        total = len(students_df)
        present_ids = attendance_df['enrollment_id'].tolist() if not attendance_df.empty else []
        present = len(present_ids)
        absent = total - present
        pct = int((present / total * 100)) if total > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", total, "🎓")
        c2.metric("Present Today", present, "✅")
        c3.metric("Absent", absent, "❌", delta_color="inverse")
        c4.metric("Avg Attendance", f"{pct}%", "📊")

        st.markdown("---")
        col_present, col_absent = st.columns(2)

        with col_present:
            st.subheader("✅ Recent Arrivals")
            recent_logs = run_query("""
                SELECT students.name, students.department, attendance.time 
                FROM attendance 
                JOIN students ON attendance.enrollment_id = students.enrollment_id
                WHERE attendance.date = ? 
                ORDER BY attendance.timestamp DESC LIMIT 10
            """, (today_str,))
            
            if not recent_logs.empty: st.dataframe(recent_logs, use_container_width=True, hide_index=True)
            else: st.info("No attendance marked yet today.")

        with col_absent:
            st.subheader("❌ Absentees List")
            if not students_df.empty:
                absent_df = students_df[~students_df['enrollment_id'].isin(present_ids)]
                display_absent = absent_df[['name', 'enrollment_id', 'department']]
                if not display_absent.empty: st.dataframe(display_absent, use_container_width=True, hide_index=True)
                else: st.success("🎉 Everyone is Present!")
            else:
                st.warning("No students in database.")
        
        if not students_df.empty:
            st.subheader("📊 Department Analytics")
            df = students_df.copy()
            df['Status'] = df['enrollment_id'].apply(lambda x: 'Present' if x in present_ids else 'Absent')
            counts = df.groupby(['department', 'Status']).size().unstack(fill_value=0)
            if 'Present' not in counts.columns: counts['Present'] = 0
            if 'Absent' not in counts.columns: counts['Absent'] = 0
            counts['Attendance %'] = (counts['Present'] / (counts['Present'] + counts['Absent']) * 100).round(1)
            
            c_chart, c_data = st.columns([2, 1])
            c_chart.bar_chart(counts['Attendance %'], color="#4CAF50")
            c_data.dataframe(counts[['Present', 'Absent', 'Attendance %']], use_container_width=True)

    # ==========================================
    # PAGE 2: LIVE AI MONITOR
    # ==========================================
    elif page == "🔴 Live AI Monitor":
        st.title("🔴 High-Speed AI Monitor")
        
        if 'camera_running' not in st.session_state:
            st.session_state['camera_running'] = False

        c1, c2 = st.columns([2, 1])
        with c2:
            st.subheader("⚙️ Controls")
            cam_source = st.radio("Camera", ["DroidCam", "Laptop Webcam"])
            
            # CSS/JS Injection for Start/Stop Button Colors
            components.html(
                """
                <script>
                const observer = new MutationObserver(() => {
                    const buttons = window.parent.document.querySelectorAll('button p');
                    buttons.forEach(p => {
                        if (p.innerText.includes('Start Camera')) {
                            p.closest('button').style.backgroundColor = '#28a745';
                            p.closest('button').style.color = 'white';
                            p.closest('button').style.border = 'none';
                        }
                        if (p.innerText.includes('Stop Camera')) {
                            p.closest('button').style.backgroundColor = '#dc3545';
                            p.closest('button').style.color = 'white';
                            p.closest('button').style.border = 'none';
                        }
                    });
                });
                observer.observe(window.parent.document.body, { childList: true, subtree: true });
                </script>
                """,
                height=0
            )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                if st.button("▶ Start Camera", use_container_width=True):
                    st.session_state['camera_running'] = True
                    st.rerun()
            with btn_col2:
                if st.button("⏹ Stop Camera", use_container_width=True):
                    st.session_state['camera_running'] = False
                    st.rerun()

            st.divider()
            st.subheader("📸 Last Scan")
            scan_img_placeholder = st.empty()
            scan_name_placeholder = st.empty()
            st.divider()
            log_placeholder = st.empty()

        with c1:
            video_placeholder = st.empty()

        if st.session_state['camera_running']:
            if not os.path.exists(ENCODING_FILE):
                st.error("⚠️ Encodings missing! Register students first.")
            else:
                try:
                    data = pickle.loads(open(ENCODING_FILE, "rb").read())
                    known_encodings, known_names = data["encodings"], data["names"]

                    student_map = run_query("SELECT enrollment_id, name FROM students").set_index('enrollment_id')['name'].to_dict()

                    src = 0 if cam_source == "Laptop Webcam" else DEFAULT_DROIDCAM
                    cap = cv2.VideoCapture(src)
                    
                    last_save_time = 0
                    
                    while st.session_state['camera_running']:
                        ret, frame = cap.read()
                        if not ret: break
                        
                        h, w, _ = frame.shape
                        detector.setInputSize((w, h))
                        _, faces = detector.detect(frame)

                        if faces is not None:
                            for face in faces:
                                aligned = recognizer.alignCrop(frame, face)
                                feat = recognizer.feature(aligned)
                                
                                identity_id = "Unknown"
                                max_score = 0
                                color = (0, 0, 255)
                                
                                for i, known_feat in enumerate(known_encodings):
                                    score = recognizer.match(feat, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                                    if score > 0.40 and score > max_score:
                                        max_score = score
                                        identity_id = known_names[i]

                                display_name = student_map.get(identity_id, identity_id) if identity_id != "Unknown" else "Unknown"
                                
                                if identity_id != "Unknown":
                                    color = (0, 255, 0)
                                    conn = get_connection()
                                    now = datetime.now()
                                    pattern = f"{now.strftime('%Y-%m-%d')} {now.strftime('%H')}:%"
                                    if not conn.execute("SELECT * FROM attendance WHERE enrollment_id=? AND timestamp LIKE ?", (identity_id, pattern)).fetchone():
                                        conn.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", 
                                                     (identity_id, now.strftime('%Y-%m-%d %H:%M:%S'), now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')))
                                        conn.commit()
                                        log_placeholder.success(f"✅ Marked: {display_name}")
                                    conn.close()
                                
                                else:
                                    current_time = time.time()
                                    if current_time - last_save_time > 5:
                                        now = datetime.now()
                                        date_str = now.strftime('%Y-%m-%d')
                                        time_str = now.strftime('%H-%M-%S')
                                        filename = f"{INTRUDER_DIR}/intruder_{CAMERA_LOCATION}_{date_str}_{time_str}.jpg"
                                        cv2.imwrite(filename, frame) 
                                        last_save_time = current_time
                                        st.toast("🚨 Intruder Saved!", icon="📸")

                                coords = face[:-1].astype(int)
                                cv2.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color, 2)
                                cv2.rectangle(frame, (coords[0], coords[1]-35), (coords[0]+coords[2], coords[1]), color, cv2.FILLED)
                                cv2.putText(frame, display_name, (coords[0]+6, coords[1]-6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

                                face_crop = frame[max(0, coords[1]-20):min(coords[1]+coords[3]+20, h), max(0, coords[0]-20):min(coords[0]+coords[2]+20, w)]
                                if face_crop.size > 0:
                                    scan_img_placeholder.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), use_container_width=True)
                                    if display_name == "Unknown": scan_name_placeholder.error("🚨 STRANGER!")
                                    else: scan_name_placeholder.success(f"✅ {display_name}")

                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

                    cap.release()
                except Exception as e: st.error(f"Error: {e}")

    # ==========================================
    # PAGE 3: INTRUDER LOGS
    # ==========================================
    elif page == "🚨 Intruder Logs":
        st.title("🚨 Intruder Activity Logs")
        
        if os.path.exists(INTRUDER_DIR):
            files = [f for f in os.listdir(INTRUDER_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not files: 
                st.info("✅ No intruder activity detected.")
            else:
                # Sort newest first
                files.sort(key=lambda x: os.path.getmtime(os.path.join(INTRUDER_DIR, x)), reverse=True)
                
                # --- 1. PRE-PARSE METADATA FOR SMART DROPDOWN ---
                file_metadata = {}
                unique_dates = set()
                
                for file in files:
                    disp_date = "Unknown Date"
                    disp_time = "Unknown Time"
                    disp_loc = "Unknown Location"
                    
                    name_parts = file.replace(".jpg", "").replace(".png", "").split("_")
                    
                    try:
                        if file.startswith("intruder_"):
                            disp_loc = name_parts[1].replace("-", " ")
                            disp_date = name_parts[2]
                            disp_time = name_parts[3].replace("-", ":")
                        elif file.startswith("unknown_"):
                            disp_loc = "Default Camera"
                            raw_date = name_parts[1]
                            raw_time = name_parts[2]
                            disp_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
                            disp_time = f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:]}"
                    except Exception:
                        pass 
                        
                    file_metadata[file] = {
                        "date": disp_date, 
                        "time": disp_time, 
                        "loc": disp_loc
                    }
                    unique_dates.add(disp_date)
                
                # Sort dates latest to oldest
                sorted_dates = sorted(list(unique_dates), reverse=True)
                
                # --- 2. PROFESSIONAL UI: SEARCH & BULK DELETE ---
                st.markdown("### 🔍 Search & Filter")
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    # Smart dropdown instead of a checkbox
                    selected_date = st.selectbox("Select Date to View", ["View All"] + sorted_dates)
                
                with c2:
                    # Only show bulk delete if a specific date is selected
                    if selected_date != "View All":
                        with st.expander(f"⚠️ Bulk Delete Logs for {selected_date}"):
                            st.warning("This action cannot be undone. All intruder photos for this date will be permanently deleted.")
                            
                            # Safety lock
                            confirm_text = st.text_input("Type 'CONFIRM' to unlock", key="confirm_bulk_del")
                            
                            if st.button("🔥 Purge Date Logs", type="primary", disabled=(confirm_text != "CONFIRM")):
                                delete_count = 0
                                for file, meta in file_metadata.items():
                                    if meta["date"] == selected_date:
                                        os.remove(os.path.join(INTRUDER_DIR, file))
                                        delete_count += 1
                                        
                                st.success(f"Successfully deleted {delete_count} logs for {selected_date}.")
                                time.sleep(1.5)
                                st.rerun()
                                
                st.markdown("---")
                
                # --- 3. DISPLAY THE LOGS ---
                cols = st.columns(3) 
                display_count = 0
                
                for file in files:
                    meta = file_metadata[file]
                    
                    # Apply filter
                    if selected_date != "View All" and meta["date"] != selected_date:
                        continue 
                        
                    with cols[display_count % 3]:
                        file_path = os.path.join(INTRUDER_DIR, file)
                        st.image(file_path, use_container_width=True)
                        st.markdown(f"**📅 Date:** `{meta['date']}`")
                        st.markdown(f"**⏰ Time:** `{meta['time']}`")
                        st.markdown(f"**📍 Loc:** `{meta['loc']}`")
                        
                        if st.button("🗑️ Delete Record", key=f"del_{file}"):
                            os.remove(file_path)
                            st.rerun()
                            
                    display_count += 1
                
                if display_count == 0:
                    st.warning(f"No intruders found for {selected_date}.")
    # ==========================================
    # PAGE 4: PERIOD MATRIX
    # ==========================================
    elif page == "📊 Period Matrix":
        st.title("📊 Period Matrix")
        c1, c2 = st.columns([1, 3])
        with c1: view_date_str = st.date_input("Date").strftime('%Y-%m-%d')
        with c2: st.info(f"Schedule: {', '.join(PERIODS)}")

        students = run_query("SELECT enrollment_id, name FROM students")
        logs = run_query("SELECT enrollment_id, timestamp FROM attendance WHERE date=?", (view_date_str,))
        
        att_map = {sid: [] for sid in students['enrollment_id']}
        for _, row in logs.iterrows():
            pid = map_timestamp_to_period(row['timestamp'])
            if pid and row['enrollment_id'] in att_map: att_map[row['enrollment_id']].append(pid)

        data = []
        if not students.empty:
            for _, row in students.iterrows():
                d = {"ID": row['enrollment_id'], "Name": row['name']}
                for p in PERIODS: d[p] = "P" if p in att_map[row['enrollment_id']] else "A"
                data.append(d)
                
            df_matrix = pd.DataFrame(data)
            
            st.markdown("### 📥 Download Matrix")
            dl1, dl2, dl3 = st.columns(3)
            
            dl1.download_button("📄 Download CSV", df_matrix.to_csv(index=False).encode('utf-8'), f"PeriodMatrix_{view_date_str}.csv", "text/csv", use_container_width=True)
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df_matrix.to_excel(writer, index=False, sheet_name='Daily Matrix')
            dl2.download_button("📊 Download Excel", excel_buffer.getvalue(), f"PeriodMatrix_{view_date_str}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            
            pdf_data = generate_period_pdf_report(df_matrix, view_date_str)
            dl3.download_button("📕 Download PDF", pdf_data, f"PeriodMatrix_{view_date_str}.pdf", "application/pdf", use_container_width=True)
            
            st.markdown("---")

            cols = st.columns([1.5, 2] + [1]*5)
            cols[0].write("**ID**"); cols[1].write("**Name**")
            for i, p in enumerate(PERIODS): cols[i+2].write(f"**{p}**")
            
            for _, row in students.iterrows():
                sid = row['enrollment_id']
                cols = st.columns([1.5, 2] + [1]*5)
                cols[0].write(sid); cols[1].write(row['name'])
                for i, p in enumerate(PERIODS):
                    present = p in att_map[sid]
                    if cols[i+2].button("✅" if present else "❌", key=f"{sid}_{p}"):
                        conn = get_connection()
                        if present:
                            conn.execute("DELETE FROM attendance WHERE enrollment_id=? AND date=? AND time BETWEEN ? AND ?", 
                                         (sid, view_date_str, SCHEDULE_MAP[p][0].strftime('%H:%M:%S'), SCHEDULE_MAP[p][1].strftime('%H:%M:%S')))
                        else:
                            start = SCHEDULE_MAP[p][0]
                            conn.execute("INSERT INTO attendance VALUES (?, ?, ?, ?)", 
                                         (sid, f"{view_date_str} {start}", view_date_str, start.strftime('%H:%M:%S')))
                        conn.commit(); conn.close()
                        st.rerun()

    # ==========================================
    # PAGE 5: ATTENDANCE REPORTS (MONTHLY & CUSTOM)
    # ==========================================
    elif page == "📅 Monthly Reports": 
        st.title("📅 Attendance Reports")
        
        tab1, tab2 = st.tabs(["📆 Monthly Report", "🗓️ Custom Date Range"])
        
        # ------------------------------------------
        # TAB 1: THE ORIGINAL MONTHLY GENERATOR
        # ------------------------------------------
        with tab1:
            st.subheader("Monthly Attendance Generator")
            
            c1, c2, c3 = st.columns(3)
            with c1: 
                sel_month = st.selectbox("Select Month", range(1, 13), format_func=lambda x: datetime(2000, x, 1).strftime('%B'))
            with c2: 
                sel_year = st.selectbox("Select Year", range(datetime.now().year - 2, datetime.now().year + 1), index=2)
            with c3:
                working_days_m = st.number_input("Total Working Days", min_value=1, max_value=31, value=20, key="wd_m")
                
            month_str = f"{sel_year}-{sel_month:02d}"
            st.info(f"Generating report for: **{datetime(sel_year, sel_month, 1).strftime('%B %Y')}**")
            
            df_report_m = run_query("SELECT enrollment_id as ID, name as Name, department as Dept FROM students ORDER BY ID")
            
            if df_report_m.empty:
                st.warning("No students found in the database.")
            else:
                df_report_m['Days_Present'] = 0
                df_report_m['Total_Periods'] = 0
                df_report_m['Total_Hours'] = 0
                
                att_df_m = run_query("SELECT enrollment_id as ID, date, timestamp FROM attendance WHERE date LIKE ?", (f"{month_str}%",))
                
                if not att_df_m.empty:
                    att_df_m['Hour_Marker'] = att_df_m['timestamp'].str[:13] 
                    total_hours_m = att_df_m.groupby('ID')['Hour_Marker'].nunique().reset_index()
                    total_hours_m.rename(columns={'Hour_Marker': 'Total_Hours'}, inplace=True)
                    
                    att_df_m['Period'] = att_df_m['timestamp'].apply(map_timestamp_to_period)
                    valid_att_m = att_df_m.dropna(subset=['Period']).copy()
                    
                    if not valid_att_m.empty:
                        valid_att_m['Day_Period'] = valid_att_m['date'] + "_" + valid_att_m['Period']
                        
                        days_present_m = valid_att_m.groupby('ID')['date'].nunique().reset_index()
                        days_present_m.rename(columns={'date': 'Days_Present'}, inplace=True)
                        
                        total_periods_m = valid_att_m.groupby('ID')['Day_Period'].nunique().reset_index()
                        total_periods_m.rename(columns={'Day_Period': 'Total_Periods'}, inplace=True)
                        
                        df_report_m = df_report_m.drop(columns=['Days_Present', 'Total_Periods', 'Total_Hours'])
                        df_report_m = df_report_m.merge(days_present_m, on='ID', how='left')
                        df_report_m = df_report_m.merge(total_periods_m, on='ID', how='left')
                        df_report_m = df_report_m.merge(total_hours_m, on='ID', how='left')
                        df_report_m.fillna(0, inplace=True)
                
                df_report_m['Days_Present'] = df_report_m['Days_Present'].astype(int)
                df_report_m['Total_Periods'] = df_report_m['Total_Periods'].astype(int)
                df_report_m['Total_Hours'] = df_report_m['Total_Hours'].astype(int) 
                
                # --- NEW: CALCULATE BOTH PERCENTAGES ---
                df_report_m['Day_%'] = ((df_report_m['Days_Present'] / working_days_m) * 100).round(1).astype(str) + '%'
                
                total_possible_periods_m = working_days_m * len(PERIODS)
                df_report_m['Period_%'] = ((df_report_m['Total_Periods'] / total_possible_periods_m) * 100).round(1).astype(str) + '%'
                # ---------------------------------------
                
                st.dataframe(df_report_m, use_container_width=True, hide_index=True)
                
                st.markdown("### 📥 Download Options")
                dl1, dl2, dl3 = st.columns(3)
                
                csv_data_m = df_report_m.to_csv(index=False).encode('utf-8')
                dl1.download_button("📄 Download CSV", csv_data_m, f"Attendance_{month_str}.csv", "text/csv", use_container_width=True, key="dl_csv_m")
                
                excel_buffer_m = io.BytesIO()
                with pd.ExcelWriter(excel_buffer_m, engine='openpyxl') as writer:
                    df_report_m.to_excel(writer, index=False, sheet_name='Monthly Report')
                dl2.download_button("📊 Download Excel", excel_buffer_m.getvalue(), f"Attendance_{month_str}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_xl_m")
                
                pdf_data_m = generate_pdf_report(df_report_m, month_str)
                dl3.download_button("📕 Download PDF", pdf_data_m, f"Attendance_{month_str}.pdf", "application/pdf", use_container_width=True, key="dl_pdf_m")

                st.markdown("---")
                st.subheader("🔍 View Detailed Monthly Log")
                student_dict_m = dict(zip(df_report_m['Name'] + " (" + df_report_m['ID'] + ")", df_report_m['ID']))
                sel_stu_m = st.selectbox("Select Student", ["Select..."] + list(student_dict_m.keys()), key="det_sel_m")
                
                if sel_stu_m != "Select...":
                    sid_m = student_dict_m[sel_stu_m]
                    det_logs_m = run_query("SELECT date as Date, time as Time, timestamp FROM attendance WHERE enrollment_id=? AND date LIKE ? ORDER BY timestamp DESC", (sid_m, f"{month_str}%"))
                    if not det_logs_m.empty:
                        det_logs_m['Period'] = det_logs_m['timestamp'].apply(map_timestamp_to_period)
                        det_logs_m['Period'] = det_logs_m['Period'].fillna("❌ Outside Schedule")
                        st.dataframe(det_logs_m[['Date', 'Time', 'Period']], use_container_width=True, hide_index=True)
                    else:
                        st.info("No recorded logs for this student in the selected month.")

        # ------------------------------------------
        # TAB 2: THE CUSTOM DATE RANGE GENERATOR
        # ------------------------------------------
        with tab2:
            st.subheader("Custom Range Attendance Generator")
            
            c1, c2 = st.columns([2, 1])
            with c1: 
                today = datetime.now().date()
                first_day = today.replace(day=1)
                date_range = st.date_input("Select Date Range", value=(first_day, today))
                
            with c2:
                working_days_r = st.number_input("Total Working Days in Range", min_value=1, max_value=200, value=20, key="wd_r")
                
            if len(date_range) != 2:
                st.warning("⚠️ Please select both a Start Date and an End Date from the calendar.")
            else:
                start_date, end_date = date_range
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                report_label = f"{start_str} to {end_str}"
                
                st.info(f"Generating report from: **{start_str}** to **{end_str}**")
                
                df_report_r = run_query("SELECT enrollment_id as ID, name as Name, department as Dept FROM students ORDER BY ID")
                
                if df_report_r.empty:
                    st.warning("No students found in the database.")
                else:
                    df_report_r['Days_Present'] = 0
                    df_report_r['Total_Periods'] = 0
                    df_report_r['Total_Hours'] = 0
                    
                    att_df_r = run_query("SELECT enrollment_id as ID, date, timestamp FROM attendance WHERE date BETWEEN ? AND ?", (start_str, end_str))
                    
                    if not att_df_r.empty:
                        att_df_r['Hour_Marker'] = att_df_r['timestamp'].str[:13]
                        total_hours_r = att_df_r.groupby('ID')['Hour_Marker'].nunique().reset_index()
                        total_hours_r.rename(columns={'Hour_Marker': 'Total_Hours'}, inplace=True)
                        
                        att_df_r['Period'] = att_df_r['timestamp'].apply(map_timestamp_to_period)
                        valid_att_r = att_df_r.dropna(subset=['Period']).copy()
                        
                        if not valid_att_r.empty:
                            valid_att_r['Day_Period'] = valid_att_r['date'] + "_" + valid_att_r['Period']
                            
                            days_present_r = valid_att_r.groupby('ID')['date'].nunique().reset_index()
                            days_present_r.rename(columns={'date': 'Days_Present'}, inplace=True)
                            
                            total_periods_r = valid_att_r.groupby('ID')['Day_Period'].nunique().reset_index()
                            total_periods_r.rename(columns={'Day_Period': 'Total_Periods'}, inplace=True)
                            
                            df_report_r = df_report_r.drop(columns=['Days_Present', 'Total_Periods', 'Total_Hours'])
                            df_report_r = df_report_r.merge(days_present_r, on='ID', how='left')
                            df_report_r = df_report_r.merge(total_periods_r, on='ID', how='left')
                            df_report_r = df_report_r.merge(total_hours_r, on='ID', how='left')
                            df_report_r.fillna(0, inplace=True)
                    
                    df_report_r['Days_Present'] = df_report_r['Days_Present'].astype(int)
                    df_report_r['Total_Periods'] = df_report_r['Total_Periods'].astype(int)
                    df_report_r['Total_Hours'] = df_report_r['Total_Hours'].astype(int) 
                    
                    # --- NEW: CALCULATE BOTH PERCENTAGES ---
                    df_report_r['Day_%'] = ((df_report_r['Days_Present'] / working_days_r) * 100).round(1).astype(str) + '%'
                    
                    total_possible_periods_r = working_days_r * len(PERIODS)
                    df_report_r['Period_%'] = ((df_report_r['Total_Periods'] / total_possible_periods_r) * 100).round(1).astype(str) + '%'
                    # ---------------------------------------
                    
                    st.dataframe(df_report_r, use_container_width=True, hide_index=True)
                    
                    st.markdown("### 📥 Download Options")
                    dl1, dl2, dl3 = st.columns(3)
                    
                    safe_label = f"{start_str}_to_{end_str}"
                    
                    csv_data_r = df_report_r.to_csv(index=False).encode('utf-8')
                    dl1.download_button("📄 Download CSV", csv_data_r, f"Attendance_{safe_label}.csv", "text/csv", use_container_width=True, key="dl_csv_r")
                    
                    excel_buffer_r = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer_r, engine='openpyxl') as writer:
                        df_report_r.to_excel(writer, index=False, sheet_name='Range Report')
                    dl2.download_button("📊 Download Excel", excel_buffer_r.getvalue(), f"Attendance_{safe_label}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True, key="dl_xl_r")
                    
                    pdf_data_r = generate_pdf_report(df_report_r, report_label)
                    dl3.download_button("📕 Download PDF", pdf_data_r, f"Attendance_{safe_label}.pdf", "application/pdf", use_container_width=True, key="dl_pdf_r")

                    st.markdown("---")
                    st.subheader("🔍 View Detailed Log (Date Range)")
                    student_dict_r = dict(zip(df_report_r['Name'] + " (" + df_report_r['ID'] + ")", df_report_r['ID']))
                    sel_stu_r = st.selectbox("Select Student", ["Select..."] + list(student_dict_r.keys()), key="det_sel_r")
                    
                    if sel_stu_r != "Select...":
                        sid_r = student_dict_r[sel_stu_r]
                        det_logs_r = run_query("SELECT date as Date, time as Time, timestamp FROM attendance WHERE enrollment_id=? AND date BETWEEN ? AND ? ORDER BY timestamp DESC", (sid_r, start_str, end_str))
                        if not det_logs_r.empty:
                            det_logs_r['Period'] = det_logs_r['timestamp'].apply(map_timestamp_to_period)
                            det_logs_r['Period'] = det_logs_r['Period'].fillna("❌ Outside Schedule")
                            st.dataframe(det_logs_r[['Date', 'Time', 'Period']], use_container_width=True, hide_index=True)
                        else:
                            st.info("No recorded logs for this student in the selected date range.")
    # ==========================================
    # PAGE 6: REGISTRATION (WITH DROPDOWNS)
    # ==========================================
    elif page == "📝 Register Student":
        st.title("📝 Multi-Angle Registration")
        
        # --- NEW: Define your College's Departments and Courses here ---
        DEPARTMENTS = ["Computer Science", "Information Technology", "Mechanical", "Civil", "Electronics", "Electrical", "Other"]
        COURSES = ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc", "M.Sc", "Other"]
        
        if "reg_form_key" not in st.session_state:
            st.session_state["reg_form_key"] = 0
            
        with st.form(f"reg_form_{st.session_state['reg_form_key']}", clear_on_submit=False):
            c1, c2 = st.columns(2)
            id_val = c1.text_input("ID")
            name_val = c2.text_input("Name")
            
            c3, c4 = st.columns(2)
            adm_val = c3.text_input("Adm No")
            # --- FIXED: Changed to Selectbox ---
            dept_val = c4.selectbox("Dept", ["Select..."] + DEPARTMENTS)
            
            c5, c6 = st.columns(2)
            # --- FIXED: Changed to Selectbox ---
            course_val = c5.selectbox("Course", ["Select..."] + COURSES)
            
            # (Optional: You could also do this for Year if you want!)
            year_val = c6.selectbox("Year", ["Select...", "1st Year", "2nd Year", "3rd Year", "4th Year"])
            
            cam_source = st.radio("Camera", ["DroidCam", "Webcam"], horizontal=True)
            sub = st.form_submit_button("📸 Start Capture Process")

        if sub:
            # --- FIXED: Added validation to make sure they actually picked something from the dropdown ---
            if not id_val or not name_val or not adm_val or dept_val == "Select..." or course_val == "Select..." or year_val == "Select...": 
                st.error("⚠️ All fields (ID, Name, Adm No, Dept, Course, and Year) are required!")
            else:
                conn = get_connection()
                
                if conn.execute("SELECT * FROM students WHERE enrollment_id=?", (id_val,)).fetchone():
                    st.error(f"⚠️ Enrollment ID '{id_val}' is already registered!")
                elif conn.execute("SELECT * FROM students WHERE admission_number=?", (adm_val,)).fetchone():
                    st.error(f"⚠️ Admission Number '{adm_val}' is already assigned!")
                else:
                    src = DEFAULT_DROIDCAM if cam_source == "DroidCam" else 0
                    cap = cv2.VideoCapture(src)
                    
                    student_dir = os.path.join(BASE_PATH, id_val)
                    os.makedirs(student_dir, exist_ok=True)
                    
                    st.info("A popup window has opened. Follow the on-screen instructions.")
                    angles = ["front", "left", "right"]
                    success_count = 0
                    
                    duplicate_found = False
                    duplicate_id = ""
                    
                    for angle in angles:
                        if duplicate_found: break 
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret: break
                            
                            display = frame.copy()
                            h, w, _ = frame.shape
                            detector.setInputSize((w, h))
                            _, faces = detector.detect(frame)
                            
                            status = f"Pose for {angle.upper()} & Press 's'"
                            if faces is not None:
                                coords = faces[0][:-1].astype(int)
                                cv2.rectangle(display, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
                            else:
                                status = "No face detected!"
                                
                            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            cv2.imshow("Registration", display)
                            
                            key = cv2.waitKey(1)
                            if key == ord('s') and faces is not None:
                                
                                if angle == "front" and os.path.exists(ENCODING_FILE):
                                    data = pickle.loads(open(ENCODING_FILE, "rb").read())
                                    if len(data["encodings"]) > 0:
                                        aligned = recognizer.alignCrop(frame, faces[0])
                                        feat = recognizer.feature(aligned)
                                        
                                        for i, known_feat in enumerate(data["encodings"]):
                                            score = recognizer.match(feat, known_feat, cv2.FaceRecognizerSF_FR_COSINE)
                                            if score > 0.40: 
                                                duplicate_found = True
                                                duplicate_id = data["names"][i]
                                                break
                                
                                if duplicate_found:
                                    break 
                                
                                cv2.imwrite(os.path.join(student_dir, f"{id_val}_{angle}.jpg"), frame)
                                success_count += 1
                                break
                            elif key == ord('q'): 
                                break
                                
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    if duplicate_found:
                        shutil.rmtree(student_dir) 
                        existing_name = conn.execute("SELECT name FROM students WHERE enrollment_id=?", (duplicate_id,)).fetchone()
                        display_name = existing_name[0] if existing_name else duplicate_id
                        st.error(f"🚨 **Registration Blocked!** This face is already registered in the system under: **{display_name} ({duplicate_id})**")
                    
                    elif success_count == 3:
                        conn.execute("INSERT INTO students VALUES (?, ?, ?, ?, ?, ?)", 
                                     (id_val, adm_val, name_val, dept_val, course_val, year_val))
                        conn.commit()
                        retrain_sface_brain() 
                        
                        st.success(f"✨ Registration complete for {name_val}!")
                        st.balloons()
                        time.sleep(2) 
                        
                        st.session_state["reg_form_key"] += 1
                        st.rerun()
                    else:
                        st.warning("⚠️ Registration incomplete. Please try again.")
                conn.close()
    # ==========================================
    # PAGE 7: DATABASE MANAGEMENT
    # ==========================================
    elif page == "📂 Database":
        st.title("📂 Database Management")
        
        df = run_query("SELECT * FROM students")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        ids = df['enrollment_id'].tolist() if not df.empty else []
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✏️ Edit & View Student")
            to_edit = st.selectbox("Select ID to View/Edit", ["Select..."] + ids, key="edit_select")
            
            if to_edit != "Select...":
                curr_data = run_query("SELECT * FROM students WHERE enrollment_id=?", (to_edit,)).iloc[0]
                
                st.markdown("**📸 Registered Faces**")
                student_dir = os.path.join(BASE_PATH, to_edit)
                
                if os.path.exists(student_dir):
                    img_cols = st.columns(3)
                    angles = ["front", "left", "right"]
                    for i, angle in enumerate(angles):
                        img_path = os.path.join(student_dir, f"{to_edit}_{angle}.jpg")
                        if os.path.exists(img_path):
                            img_cols[i].image(img_path, caption=angle.capitalize(), use_container_width=True)
                        else:
                            img_cols[i].warning(f"No {angle} img")
                else:
                    st.error("⚠️ Photos missing from server!")
                
                with st.form("edit_form"):
                    st.info(f"Editing details for ID: **{to_edit}**")
                    e_name = st.text_input("Name", curr_data['name'])
                    
                    c1, c2 = st.columns(2)
                    e_adm = c1.text_input("Admission No", curr_data['admission_number'])
                    e_dept = c2.text_input("Department", curr_data['department'])
                    
                    c3, c4 = st.columns(2)
                    e_course = c3.text_input("Course", curr_data['course'])
                    e_year = c4.text_input("Year", curr_data['year'])
                    
                    if st.form_submit_button("💾 Save Changes"):
                        conn = get_connection()
                        conn.execute('''UPDATE students 
                                        SET name=?, admission_number=?, department=?, course=?, year=? 
                                        WHERE enrollment_id=?''', 
                                     (e_name, e_adm, e_dept, e_course, e_year, to_edit))
                        conn.commit()
                        conn.close()
                        st.success(f"Successfully updated {e_name}!")
                        time.sleep(1)
                        st.rerun()

        with col2:
            st.subheader("🗑️ Delete Single Student")
            to_del = st.selectbox("Select ID to Delete", ["Select..."] + ids, key="del_select")
            
            if st.button("🚨 Permanently Delete", type="primary"):
                if to_del != "Select...":
                    conn = get_connection()
                    conn.execute("DELETE FROM students WHERE enrollment_id=?", (to_del,))
                    conn.execute("DELETE FROM attendance WHERE enrollment_id=?", (to_del,))
                    conn.commit()
                    conn.close()
                    
                    del_path = os.path.join(BASE_PATH, to_del)
                    if os.path.exists(del_path): shutil.rmtree(del_path)
                    
                    retrain_sface_brain() 
                    st.success(f"ID {to_del} deleted completely.")
                    time.sleep(1)
                    st.rerun()

        st.markdown("---")
        
        st.subheader("🎓 End of Academic Year Cleanup")
        with st.expander("⚠️ Open Batch Deletion Tool (DANGER)"):
            st.warning("This action will permanently delete student records, attendance logs, and facial recognition data for an entire Year/Batch.")
            
            unique_years = df['year'].dropna().unique().tolist() if not df.empty else []
            
            if unique_years:
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    batch_to_clear = st.selectbox("Select Batch/Year to Remove", ["Select..."] + unique_years)
                with b_col2:
                    confirm_text = st.text_input("Type 'CONFIRM' to unlock deletion")
                
                if st.button("🔥 Purge Entire Batch", type="primary", disabled=(confirm_text != "CONFIRM" or batch_to_clear == "Select...")):
                    
                    graduating_ids = run_query("SELECT enrollment_id FROM students WHERE year=?", (batch_to_clear,))['enrollment_id'].tolist()
                    
                    if graduating_ids:
                        conn = get_connection()
                        
                        conn.execute("DELETE FROM students WHERE year=?", (batch_to_clear,))
                        for gid in graduating_ids:
                            conn.execute("DELETE FROM attendance WHERE enrollment_id=?", (gid,))
                            del_path = os.path.join(BASE_PATH, gid)
                            if os.path.exists(del_path): 
                                shutil.rmtree(del_path)
                        
                        conn.commit()
                        conn.close()
                        
                        retrain_sface_brain()
                        st.success(f"Successfully purged {len(graduating_ids)} students from batch: {batch_to_clear}.")
                        time.sleep(2)
                        st.rerun()
            else:
                st.info("No batches found to clear.")