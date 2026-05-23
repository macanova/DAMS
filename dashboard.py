import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import sqlite3
import pandas as pd
from datetime import datetime
import os
import cv2
from PIL import Image, ImageTk

# Ensure directories exist
if not os.path.exists("images"):
    os.makedirs("images")

class AttendanceManager:
    def __init__(self, root):
        self.root = root
        self.root.title("DAMS - Super Admin Dashboard")
        self.root.geometry("1100x700")

        # --- SETUP DATABASE ---
        self.init_db()

        # --- TABS LAYOUT ---
        self.tabs = ttk.Notebook(root)
        self.tabs.pack(expand=1, fill="both")

        self.tab_attendance = ttk.Frame(self.tabs)
        self.tab_students = ttk.Frame(self.tabs)
        self.tab_register = ttk.Frame(self.tabs)  # <--- NEW TAB

        self.tabs.add(self.tab_attendance, text=" 📅 Daily Attendance Log ")
        self.tabs.add(self.tab_students, text=" 🎓 Student Database ")
        self.tabs.add(self.tab_register, text=" ➕ Register New Student ")

        self.setup_attendance_tab()
        self.setup_student_tab()
        self.setup_register_tab()

    def init_db(self):
        """Creates the necessary tables if they don't exist"""
        conn = sqlite3.connect('student_database.db')
        cursor = conn.cursor()
        
        # Table 1: Students (Details)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                enrollment_id TEXT PRIMARY KEY,
                admission_number TEXT,
                name TEXT,
                department TEXT,
                course TEXT,
                year TEXT
            )
        ''')
        
        # Table 2: Attendance (Logs)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                enrollment_id TEXT,
                time TEXT,
                date TEXT
            )
        ''')
        conn.commit()
        conn.close()

    # ==========================================
    # TAB 1: DAILY ATTENDANCE
    # ==========================================
    def setup_attendance_tab(self):
        frame_controls = tk.Frame(self.tab_attendance, pady=15, bg="#f0f0f0")
        frame_controls.pack(fill=tk.X)

        tk.Label(frame_controls, text="Select Date:", bg="#f0f0f0", font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        self.date_entry = tk.Entry(frame_controls, width=12, font=("Arial", 10))
        self.date_entry.insert(0, today_str)
        self.date_entry.pack(side=tk.LEFT, padx=5)

        btn_load = tk.Button(frame_controls, text="🔍 View Log", command=self.load_attendance, bg="#2196F3", fg="white", font=("Arial", 9, "bold"))
        btn_load.pack(side=tk.LEFT, padx=10)

        btn_export = tk.Button(frame_controls, text="💾 Export Excel", command=self.export_attendance, bg="#4CAF50", fg="white", font=("Arial", 9, "bold"))
        btn_export.pack(side=tk.LEFT, padx=10)

        # Columns
        cols = ("Enrollment ID", "Name", "Dept", "Year", "Time", "Status")
        self.tree_att = ttk.Treeview(self.tab_attendance, columns=cols, show="headings", height=20)
        
        for col in cols:
            self.tree_att.heading(col, text=col)
            self.tree_att.column(col, width=120, anchor="center")
        
        self.tree_att.column("Name", width=200) # Name needs more space
        self.tree_att.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def load_attendance(self):
        for item in self.tree_att.get_children(): self.tree_att.delete(item)
        target_date = self.date_entry.get()
        conn = sqlite3.connect('student_database.db')
        
        # JOIN Query to get Name/Dept from Students Table
        query = """
            SELECT students.enrollment_id, students.name, students.department, students.year, attendance.time
            FROM attendance 
            LEFT JOIN students ON attendance.enrollment_id = students.enrollment_id
            WHERE attendance.date = ? 
            ORDER BY attendance.time DESC
        """
        rows = conn.execute(query, (target_date,)).fetchall()
        
        for row in rows:
            # row = (id, name, dept, year, time)
            # If student not in DB, name might be None, so we handle that
            name = row[1] if row[1] else "Unknown (Not Registered)"
            dept = row[2] if row[2] else "-"
            year = row[3] if row[3] else "-"
            
            self.tree_att.insert("", tk.END, values=(row[0], name, dept, year, row[4], "Present"))
            
        conn.close()

    def export_attendance(self):
        target_date = self.date_entry.get()
        conn = sqlite3.connect('student_database.db')
        query = """
            SELECT students.enrollment_id, students.name, students.department, students.course, students.year, attendance.time, attendance.date
            FROM attendance 
            LEFT JOIN students ON attendance.enrollment_id = students.enrollment_id
            WHERE attendance.date = ?
        """
        try:
            df = pd.read_sql_query(query, conn, params=(target_date,))
            if df.empty:
                messagebox.showwarning("Warning", "No data found for this date.")
                return
            
            filename = filedialog.asksaveasfilename(initialfile=f"Attendance_{target_date}.csv", defaultextension=".csv")
            if filename:
                df.to_csv(filename, index=False)
                messagebox.showinfo("Success", "Export Successful!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            conn.close()

    # ==========================================
    # TAB 2: STUDENT DATABASE
    # ==========================================
    def setup_student_tab(self):
        frame_tools = tk.Frame(self.tab_students, pady=15, bg="#f0f0f0")
        frame_tools.pack(fill=tk.X)

        btn_refresh = tk.Button(frame_tools, text="🔄 Refresh List", command=self.load_students, font=("Arial", 9))
        btn_refresh.pack(side=tk.LEFT, padx=20)

        btn_del = tk.Button(frame_tools, text="❌ Delete Student", command=self.delete_student, bg="#F44336", fg="white", font=("Arial", 9, "bold"))
        btn_del.pack(side=tk.RIGHT, padx=20)

        cols = ("Enrollment ID", "Name", "Dept", "Course", "Year")
        self.tree_stu = ttk.Treeview(self.tab_students, columns=cols, show="headings", height=20)
        
        for col in cols:
            self.tree_stu.heading(col, text=col)
            self.tree_stu.column(col, anchor="center")

        self.tree_stu.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        self.load_students()

    def load_students(self):
        for item in self.tree_stu.get_children(): self.tree_stu.delete(item)
        conn = sqlite3.connect('student_database.db')
        rows = conn.execute("SELECT enrollment_id, name, department, course, year FROM students").fetchall()
        for row in rows:
            self.tree_stu.insert("", tk.END, values=row)
        conn.close()

    def delete_student(self):
        selected = self.tree_stu.selection()
        if not selected:
            messagebox.showwarning("Select Student", "Please select a student to delete.")
            return
        
        item = self.tree_stu.item(selected)
        enroll_id = item['values'][0]
        
        if messagebox.askyesno("Confirm", f"Delete student {enroll_id}? Photo will also be deleted."):
            conn = sqlite3.connect('student_database.db')
            conn.execute("DELETE FROM students WHERE enrollment_id = ?", (enroll_id,))
            conn.commit()
            conn.close()
            
            # Delete image file
            if os.path.exists(f"images/{enroll_id}.jpg"):
                os.remove(f"images/{enroll_id}.jpg")
            
            self.load_students()
            messagebox.showinfo("Done", "Student deleted.")

    # ==========================================
    # TAB 3: REGISTER NEW STUDENT (THE NEW PART)
    # ==========================================
    def setup_register_tab(self):
        frame_main = tk.Frame(self.tab_register, padx=20, pady=20)
        frame_main.pack(fill=tk.BOTH, expand=True)

        # -- Left Side: Form --
        frame_form = tk.LabelFrame(frame_main, text="Student Details", padx=10, pady=10)
        frame_form.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(frame_form, text="Enrollment ID (Must be Unique):").pack(anchor="w")
        self.entry_id = tk.Entry(frame_form); self.entry_id.pack(fill=tk.X, pady=5)

        tk.Label(frame_form, text="Full Name:").pack(anchor="w")
        self.entry_name = tk.Entry(frame_form); self.entry_name.pack(fill=tk.X, pady=5)

        tk.Label(frame_form, text="Department:").pack(anchor="w")
        self.entry_dept = tk.Entry(frame_form); self.entry_dept.pack(fill=tk.X, pady=5)

        tk.Label(frame_form, text="Course:").pack(anchor="w")
        self.entry_course = tk.Entry(frame_form); self.entry_course.pack(fill=tk.X, pady=5)

        tk.Label(frame_form, text="Year:").pack(anchor="w")
        self.entry_year = tk.Entry(frame_form); self.entry_year.pack(fill=tk.X, pady=5)

        tk.Button(frame_form, text="💾 Save Student", command=self.save_student, bg="#4CAF50", fg="white", height=2).pack(fill=tk.X, pady=20)

        # -- Right Side: Camera --
        frame_cam = tk.LabelFrame(frame_main, text="Live Camera (For Face Photo)", padx=10, pady=10)
        frame_cam.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.lbl_video = tk.Label(frame_cam)
        self.lbl_video.pack()

        tk.Button(frame_cam, text="📸 Capture Photo", command=self.capture_photo, bg="#2196F3", fg="white").pack(fill=tk.X, pady=10)
        tk.Button(frame_cam, text="▶ Start Camera", command=self.start_camera).pack(fill=tk.X)

        self.cap = None
        self.captured_image = None

    def start_camera(self):
        # NOTE: If using DroidCam, change 0 to the URL
        self.cap = cv2.VideoCapture(0) 
        self.update_video()

    def update_video(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
                self.current_frame = frame  # Keep reference for capturing
                
                # Convert to Tkinter format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.lbl_video.imgtk = imgtk
                self.lbl_video.configure(image=imgtk)
                self.lbl_video.after(10, self.update_video)

    def capture_photo(self):
        if hasattr(self, 'current_frame'):
            self.captured_image = self.current_frame
            messagebox.showinfo("Camera", "Photo Captured! Now fill details and click Save.")
        else:
            messagebox.showerror("Error", "Start camera first!")

    def save_student(self):
        enroll_id = self.entry_id.get()
        name = self.entry_name.get()
        
        if not enroll_id or not name:
            messagebox.showerror("Error", "ID and Name are required!")
            return
            
        if self.captured_image is None:
            messagebox.showerror("Error", "Please capture a photo first!")
            return

        # 1. Save Image
        cv2.imwrite(f"images/{enroll_id}.jpg", self.captured_image)

        # 2. Save DB Record
        conn = sqlite3.connect('student_database.db')
        try:
            conn.execute("""
                INSERT INTO students (enrollment_id, admission_number, name, department, course, year)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (enroll_id, "", name, self.entry_dept.get(), self.entry_course.get(), self.entry_year.get()))
            conn.commit()
            messagebox.showinfo("Success", f"Student {name} Registered!")
            
            # Clear fields
            self.entry_id.delete(0, tk.END)
            self.entry_name.delete(0, tk.END)
            self.captured_image = None
            self.load_students() # Refresh tab 2
            
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Student ID already exists!")
        finally:
            conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = AttendanceManager(root)
    root.mainloop()