import sqlite3
import pandas as pd
import os

def view_all_data():
    db_file = 'student_database.db'
    
    if not os.path.exists(db_file):
        print("❌ Error: Database file not found. Please run reset_database_v3.py first.")
        return

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    print("\n" + "="*50)
    print("   📊 DATABASE DIAGNOSTIC TOOL")
    print("="*50)

    # 1. CHECK REGISTERED STUDENTS
    print("\n--- [1] REGISTERED STUDENTS (students table) ---")
    try:
        # We select specific columns to make sure they exist
        query_students = "SELECT enrollment_id, admission_number, name, department, course, year FROM students"
        df_students = pd.read_sql_query(query_students, conn)
        
        if df_students.empty:
            print("⚠️  No students found. Run 'register.py' to add someone.")
        else:
            print(df_students)
    except Exception as e:
        print(f"❌ Error reading Students table: {e}")
        print("   (Hint: Your database might be old. Delete .db file and re-run reset script.)")

    # 2. CHECK ATTENDANCE LOGS
    print("\n--- [2] ATTENDANCE LOGS (attendance table) ---")
    try:
        query_attendance = """
            SELECT attendance.date, attendance.timestamp, students.name, students.enrollment_id 
            FROM attendance 
            LEFT JOIN students ON attendance.enrollment_id = students.enrollment_id
            ORDER BY attendance.timestamp DESC
        """
        df_att = pd.read_sql_query(query_attendance, conn)
        
        if df_att.empty:
            print("⚠️  No attendance marked yet. Run 'main.py' and look at the camera.")
        else:
            print(df_att)
    except Exception as e:
        print(f"❌ Error reading Attendance table: {e}")

    conn.close()

if __name__ == "__main__":
    view_all_data()