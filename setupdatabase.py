import sqlite3
import os

def reset_db_final():
    db_name = 'student_database.db'
    
    # 1. Remove old DB to ensure a clean slate
    if os.path.exists(db_name):
        os.remove(db_name)
        print(f"⚠️  Old {db_name} removed.")

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 2. Create Student Table (Matches your Registration Form)
    cursor.execute('''
        CREATE TABLE students (
            enrollment_id TEXT PRIMARY KEY,
            admission_number TEXT UNIQUE,  -- Removed NOT NULL to allow flexibility
            name TEXT NOT NULL,
            department TEXT,
            course TEXT,
            year TEXT
        )
    ''')

    # 3. Create Attendance Table (MATCHING THE DASHBOARD)
    # I added the 'time' column here so the dashboard doesn't crash.
    cursor.execute('''
        CREATE TABLE attendance (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment_id TEXT,
            timestamp TEXT,
            date TEXT,
            time TEXT,
            FOREIGN KEY(enrollment_id) REFERENCES students(enrollment_id)
        )
    ''')

    print("✅ Database Reset Complete!")
    print("👉 Created 'students' table with 6 columns.")
    print("👉 Created 'attendance' table with 'time' column (Critical Fix).")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    reset_db_final()