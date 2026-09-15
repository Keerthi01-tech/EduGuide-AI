from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# This 'db' object connects our Python code to the actual database file
db = SQLAlchemy()

# ---- Table 1: Users ----
class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="student")
    student_id = db.Column(db.String(50))
    course = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime)  # Powers "currently viewing"

# ---- Table 2: Prediction History ----
class PredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    probability_low = db.Column(db.Float)
    probability_medium = db.Column(db.Float)
    probability_high = db.Column(db.Float)
    top_factors = db.Column(db.String(500))
    academic_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ---- Table 3: Warnings ----
class Warning(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    reason = db.Column(db.String(500), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Remark(db.Model):
    __tablename__ = "remarks"
    id = db.Column(db.Integer, primary_key=True)
    student_user_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
 
 
class ChatMessage(db.Model):
    __tablename__ = "chatroom_messages"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # "student" or "teacher"
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
 
 
class ChatroomSettings(db.Model):
    __tablename__ = "chatroom_settings"
    id = db.Column(db.Integer, primary_key=True)
    locked = db.Column(db.Boolean, default=False)    