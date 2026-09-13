from flask import Flask, render_template, request, redirect
import joblib
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Load AI model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Create database
def create_database():
    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            problem TEXT,
            category TEXT,
            priority TEXT,
            date_time TEXT
        )
    """)

    # Add status column if it does not exist
    cursor.execute("PRAGMA table_info(tickets)")
    columns = [row[1] for row in cursor.fetchall()]

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

    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()

    # Get all tickets
    cursor.execute("SELECT * FROM tickets")
    tickets = cursor.fetchall()

    # Total tickets
    total_tickets = len(tickets)

    # Count Pending tickets
    cursor.execute("""
        SELECT COUNT(*) FROM tickets
        WHERE status = 'Pending'
    """)
    pending = cursor.fetchone()[0]

    # Count In Progress tickets
    cursor.execute("""
        SELECT COUNT(*) FROM tickets
        WHERE status = 'In Progress'
    """)
    in_progress = cursor.fetchone()[0]

    # Count Resolved tickets
    cursor.execute("""
        SELECT COUNT(*) FROM tickets
        WHERE status = 'Resolved'
    """)
    resolved = cursor.fetchone()[0]

    # Category-wise ticket count
    cursor.execute("""
        SELECT category, COUNT(*)
        FROM tickets
        GROUP BY category
    """)
    category_counts = cursor.fetchall()

    # Priority-wise ticket count
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
    ticket = request.form["ticket"]
    priority = request.form["priority"]

    # AI prediction
    ticket_vector = vectorizer.transform([ticket])
    category = model.predict(ticket_vector)[0]

    # Save ticket in database
    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tickets
        (student_name, problem, category, priority, date_time, status)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        student_name,
        ticket,
        category,
        priority,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Pending"
    ))

    conn.commit()
    conn.close()

    return f"""
    <h1>Ticket Classified Successfully!</h1>

    <p><b>Student Name:</b> {student_name}</p>
    <p><b>Problem:</b> {ticket}</p>
    <p><b>Predicted Category:</b> {category}</p>
    <p><b>Priority:</b> {priority}</p>
    <p><b>Status:</b> Pending</p>

    <br>
    <a href="/">Submit Another Ticket</a>
    """


# Update Ticket Status
@app.route("/update_status/<int:ticket_id>", methods=["POST"])
def update_status(ticket_id):

    status = request.form["status"]

    conn = sqlite3.connect("tickets.db")
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tickets
        SET status = ?
        WHERE id = ?
    """, (status, ticket_id))

    conn.commit()
    conn.close()

    return redirect("/admin")


# Run Application
create_database()

if __name__ == "__main__":
    app.run(debug=True)