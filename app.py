from flask import Flask, render_template, request, redirect
import joblib
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)

# Load AI model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "tickets.db")


# Create / Update database
def create_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create tickets table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            problem TEXT,
            category TEXT,
            priority TEXT,
            date_time TEXT,
            status TEXT DEFAULT 'Pending'
        )
    """)

    # Check existing columns
    cursor.execute("PRAGMA table_info(tickets)")
    columns = [row[1] for row in cursor.fetchall()]

    # Add department column if it does not exist
    if "department" not in columns:
        cursor.execute("""
            ALTER TABLE tickets
            ADD COLUMN department TEXT
        """)

    # Add status column if it does not exist
    if "status" not in columns:
        cursor.execute("""
            ALTER TABLE tickets
            ADD COLUMN status TEXT DEFAULT 'Pending'
        """)

    conn.commit()
    conn.close()


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Admin Dashboard
@app.route("/admin")
def admin():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all tickets in fixed order
    cursor.execute("""
        SELECT
            id,
            student_name,
            department,
            problem,
            category,
            priority,
            date_time,
            status
        FROM tickets
        ORDER BY id DESC
    """)

    tickets = cursor.fetchall()

    # Total tickets
    total_tickets = len(tickets)

    # Pending
    cursor.execute("""
        SELECT COUNT(*)
        FROM tickets
        WHERE status = 'Pending'
    """)
    pending = cursor.fetchone()[0]

    # In Progress
    cursor.execute("""
        SELECT COUNT(*)
        FROM tickets
        WHERE status = 'In Progress'
    """)
    in_progress = cursor.fetchone()[0]

    # Resolved
    cursor.execute("""
        SELECT COUNT(*)
        FROM tickets
        WHERE status = 'Resolved'
    """)
    resolved = cursor.fetchone()[0]

    # Category-wise summary
    cursor.execute("""
        SELECT category, COUNT(*)
        FROM tickets
        GROUP BY category
    """)
    category_counts = cursor.fetchall()

    # Priority-wise summary
    cursor.execute("""
        SELECT priority, COUNT(*)
        FROM tickets
        GROUP BY priority
    """)
    priority_counts = cursor.fetchall()

    conn.close()

    return render_template(
        "admin.html",
        tickets=tickets,
        total_tickets=total_tickets,
        pending=pending,
        in_progress=in_progress,
        resolved=resolved,
        category_counts=category_counts,
        priority_counts=priority_counts
    )


# Predict Ticket Category
@app.route("/predict", methods=["POST"])
def predict():

    student_name = request.form["student_name"]
    department = request.form["department"]
    ticket = request.form["ticket"]
    priority = request.form["priority"]

    # AI prediction
    ticket_vector = vectorizer.transform([ticket])
    category = model.predict(ticket_vector)[0]

    # Save ticket in database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tickets
        (
            student_name,
            department,
            problem,
            category,
            priority,
            date_time,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        department,
        ticket,
        category,
        priority,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pending"
    ))

    conn.commit()
    conn.close()

    return f"""
    <html>
    <head>
        <title>Ticket Classified Successfully</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f6f8;
                padding: 40px;
            }}

            .result {{
                max-width: 600px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 12px;
            }}

            h1 {{
                color: green;
                text-align: center;
            }}

            p {{
                font-size: 16px;
                margin: 12px 0;
            }}

            a {{
                display: inline-block;
                margin-top: 20px;
                text-decoration: none;
                font-weight: bold;
            }}
        </style>
    </head>

    <body>

    <div class="result">

        <h1>Ticket Classified Successfully!</h1>

        <p><b>Student Name:</b> {student_name}</p>

        <p><b>Department:</b> {department}</p>

        <p><b>Problem:</b> {ticket}</p>

        <p><b>Predicted Category:</b> {category}</p>

        <p><b>Priority:</b> {priority}</p>

        <p><b>Status:</b> Pending</p>

        <br>

        <a href="/">Submit Another Ticket</a>

        <br><br>

        <a href="/admin">Go to Admin Dashboard</a>

    </div>

    </body>
    </html>
    """


# Update Ticket Status
@app.route("/update_status/<int:ticket_id>", methods=["POST"])
def update_status(ticket_id):

    status = request.form["status"]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tickets
        SET status = ?
        WHERE id = ?
    """, (status, ticket_id))

    conn.commit()
    conn.close()

    return redirect("/admin")


# Initialize database
# Important for Render + Gunicorn
create_database()


# Run application locally
if __name__ == "__main__":
    app.run(debug=True)