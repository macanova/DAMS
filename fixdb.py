import sqlite3

# Connect to your database
conn = sqlite3.connect('student_database.db')
cursor = conn.cursor()

# Update ALL missing fields for Saritha (0001)
cursor.execute("""
    UPDATE students 
    SET admission_number = 'A101', 
        course = 'B.Tech', 
        year = '1st Year' 
    WHERE enrollment_id = '0001'
""")

# Update ALL missing fields for Girish (0002)
cursor.execute("""
    UPDATE students 
    SET admission_number = 'A102', 
        course = 'B.Tech', 
        year = '2nd Year' 
    WHERE enrollment_id = '0002'
""")

conn.commit()
conn.close()

print("✅ Database fully updated! Check your Streamlit Database tab again.")