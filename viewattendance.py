import sqlite3
import pandas as pd
import os

def view_attendance_log():
    # 🔴 CRITICAL FIX: Make sure we open the NEW database
    db_file = 'student_database.db' 
    
    if not os.path.exists(db_file):
        print(f"❌ Error: Database file '{db_file}' not found.")
        print("   Make sure you are in the correct folder (DAMS).")
        return

    conn = sqlite3.connect(db_file)
    
    print(f"--- READING FROM: {db_file} ---")

    # SQL Query: Join the Attendance Log with Student Details
    query = """
        SELECT 
            attendance.date, 
            attendance.timestamp, 
            students.enrollment_id, 
            students.name, 
            students.department 
        FROM attendance 
        JOIN students ON attendance.enrollment_id = students.enrollment_id
        ORDER BY attendance.timestamp DESC
    """
    
    try:
        # Use Pandas to print a pretty table
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("⚠️  Database is open, but no attendance records found yet.")
            print("   (Run main.py and look at the camera to generate logs!)")
        else:
            print("\n--- ✅ LIVE ATTENDANCE LOG ---")
            print(df)
            
    except Exception as e:
        print(f"❌ Database Error: {e}")
        print("   (This usually means the database structure is old. Try running reset_database_v3.py)")
    finally:
        conn.close()

if __name__ == "__main__":
    view_attendance_log()