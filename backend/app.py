from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from pathlib import Path
import joblib
import pandas as pd
import shap
import numpy as np
import ollama
import json
import os
from datetime import datetime, timedelta
try:
    from backend.models_db import db, User, PredictionRecord, Warning, Remark, ChatMessage, ChatroomSettings
except ImportError:
    from models_db import db, User, PredictionRecord, Warning, Remark, ChatMessage, ChatroomSettings


app = Flask(__name__)
CORS(app)
app.json.sort_keys = False

# Tell Flask where the database lives
BASE_DIR = Path(__file__).resolve().parent.parent

# Online (Vercel): use the hosted database whose address is stored as a secret setting.
# On your laptop: use the local SQLite file, as before.
DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True}
else:
    if os.environ.get("VERCEL"):
        db_path = "/tmp/eduguide.db"
    else:
        os.makedirs(BASE_DIR / "database", exist_ok=True)
        db_path = BASE_DIR / "database" / "eduguide.db"
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False


try:
    from backend.models_db import db, User, PredictionRecord, Warning, Remark, ChatMessage, ChatroomSettings
except ImportError:
    from models_db import db, User, PredictionRecord, Warning, Remark, ChatMessage, ChatroomSettings

db.init_app(app)

with app.app_context():
    db.create_all()

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PROBABILITY_MODEL_PATH = BASE_DIR / "models" / "probability_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
ENCODERS_PATH = BASE_DIR / "models" / "encoders.pkl"
model = joblib.load(MODEL_PATH)
probability_model = (
    joblib.load(PROBABILITY_MODEL_PATH)
    if PROBABILITY_MODEL_PATH.exists()
    else model
)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)
risk_encoder = encoders["risk_level"]

background_df = pd.read_csv(BASE_DIR / "data" / "processed" / "students_processed.csv")
background_X = background_df.drop(columns=["risk_level"])
background_scaled = scaler.transform(background_X)
explainer = shap.LinearExplainer(model, background_scaled)

FEATURE_COLUMNS = [
    "gender", "age", "Grade", "absence_days", "Course", "GPA", "Absent",
    "study_time", "Python_score", "Java_score", "C_score", "avg_score",
    "test_prep", "exam_score", "attendance", "hours_studied",
    "motivation_level", "internet_access", "previous_scores", "sleep_hours"
]

def validate_features(data):
    """Checks that all required fields are present and numeric."""
    missing = [col for col in FEATURE_COLUMNS if col not in data]
    if missing:
        return f"Missing required field(s): {', '.join(missing)}"

    non_numeric = [col for col in FEATURE_COLUMNS if not isinstance(data[col], (int, float))]
    if non_numeric:
        return f"Field(s) must be numbers: {', '.join(non_numeric)}"

    return None


def prepare_features(data):
    return [data[col] for col in FEATURE_COLUMNS]

    

# Assigns a severity number to each risk level, so we can compare "before vs after"
RISK_SEVERITY = {"low": 0, "medium": 1, "high": 2}

# Plain-English advice for each feature, used when that feature is contributing to risk
RECOMMENDATIONS = {
    "attendance": {
        "immediate": "Attend your next class without fail — consistency starts now.",
        "short_term": "Aim for at least 90% attendance over the next month.",
        "resource": "Talk to your academic advisor if attendance is affected by outside circumstances."
    },
    "absence_days": {
        "immediate": "Plan to attend all upcoming classes this week.",
        "short_term": "Track your attendance weekly to catch dips early.",
        "resource": "Set calendar reminders for class timings."
    },
    "GPA": {
        "immediate": "Identify the one subject pulling your GPA down the most.",
        "short_term": "Dedicate extra revision time to your weakest subject this month.",
        "resource": "Consider peer study groups or a tutor for that subject."
    },
    "avg_score": {
        "immediate": "Review your most recent test to find recurring mistakes.",
        "short_term": "Practice consistently across all subjects, not just before exams.",
        "resource": "Use practice question banks for extra repetition."
    },
    "exam_score": {
        "immediate": "Revisit the topics from your last exam that scored lowest.",
        "short_term": "Create a study schedule leading up to your next exam.",
        "resource": "Ask a teacher for feedback on your last exam paper."
    },
    "previous_scores": {
        "immediate": "Look back at where your scores started dropping.",
        "short_term": "Set a small, specific score-improvement goal for next month.",
        "resource": "Keep a simple score-tracking log to see trends."
    },
    "study_time": {
        "immediate": "Block out at least 1 extra hour of study today.",
        "short_term": "Build a consistent weekly study routine.",
        "resource": "Try a study timer method (like Pomodoro) for focus."
    },
    "hours_studied": {
        "immediate": "Add a short, focused study session today.",
        "short_term": "Gradually increase weekly study hours.",
        "resource": "Use a study planner app to stay consistent."
    },
    "Python_score": {
        "immediate": "Redo one Python exercise you struggled with recently.",
        "short_term": "Practice Python problems a few times a week.",
        "resource": "Use free platforms like HackerRank for practice."
    },
    "Java_score": {
        "immediate": "Review a recent Java concept you found confusing.",
        "short_term": "Practice Java problems a few times a week.",
        "resource": "Use free platforms like HackerRank for practice."
    },
    "C_score": {
        "immediate": "Revisit a C programming topic you found difficult recently.",
        "short_term": "Practice C problems a few times a week.",
        "resource": "Use free platforms like HackerRank for practice."
    },
    "Absent": {
        "immediate": "Make a plan to attend all classes this week.",
        "short_term": "Identify and address the main cause of absences.",
        "resource": "Speak with your instructor if you're falling behind."
    },
    "motivation_level": {
        "immediate": "Set one small, achievable task for today.",
        "short_term": "Track small wins weekly to rebuild momentum.",
        "resource": "Consider speaking with a mentor or counselor."
    },
    "test_prep": {
        "immediate": "Look into what a test-prep course near you covers.",
        "short_term": "Complete a structured test-prep course this term.",
        "resource": "Check if your institution offers free prep resources."
    },
    "internet_access": {
        "immediate": "Identify a reliable place to study with stable internet.",
        "short_term": "Plan study sessions around reliable connectivity windows.",
        "resource": "Check if your institution offers library/lab access."
    },
    "sleep_hours": {
        "immediate": "Aim for consistent sleep timing tonight.",
        "short_term": "Build a steady sleep schedule this week.",
        "resource": "Avoid screens right before bed for better sleep quality."
    },
}

@app.route("/")
def home():
    return jsonify({"message": "EduGuide AI backend is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # NEW: get which student this prediction is for
        user_id = data.get("user_id")
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400

        # NEW: validate input before using it
        validation_error = validate_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        features = prepare_features(data)
        features_scaled = scaler.transform([features])

        prediction_num = model.predict(features_scaled)[0]
        probabilities = probability_model.predict_proba(features_scaled)[0]
        risk_label = risk_encoder.inverse_transform([prediction_num])[0]

        prob_dict = {
            str(risk_encoder.inverse_transform([int(class_id)])[0]).lower():
                round(float(probability), 6)
            for class_id, probability in zip(
                probability_model.classes_,
                probabilities
            )
        }

        # NEW: save this prediction into the database
        record = PredictionRecord(
            user_id=user_id,
            risk_level=risk_label,
            probability_low=prob_dict.get("low", 0),
            probability_medium=prob_dict.get("medium", 0),
            probability_high=prob_dict.get("high", 0),
            academic_data=json.dumps({col: data[col] for col in FEATURE_COLUMNS})
        )
        db.session.add(record)
        db.session.commit()

        response = {
            "risk_level": risk_label,
            "probabilities": prob_dict,
            "record_id": record.id   # confirms it was saved, and lets you find it later
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        features = prepare_features(data)
        features_scaled = scaler.transform([features])

        prediction_num = model.predict(features_scaled)[0]
        shap_values = explainer.shap_values(features_scaled)

        if isinstance(shap_values, list):
            values_for_class = shap_values[prediction_num][0]
        else:
            values_for_class = shap_values[0, :, prediction_num]

        impact = sorted(
            zip(FEATURE_COLUMNS, values_for_class),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation = [
            {"feature": name, "impact": round(float(value), 4)}
            for name, value in impact
        ]

        risk_label = risk_encoder.inverse_transform([prediction_num])[0]

        return jsonify({
            "risk_level": risk_label,
            "top_factors": explanation[:5]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()

        if data.get("role") == "admin":
            return jsonify({"error": "Admin accounts cannot be created through public registration."}), 403

        # Check if email is already used
        existing_user = User.query.filter_by(email=data["email"]).first()
        if existing_user:
            return jsonify({"error": "Email already registered"}), 400

        # Create the scrambled password
        hashed_password = generate_password_hash(data["password"])

        # Create a new User row and save it
        new_user = User(
            name=data["name"],
            email=data["email"],
            password_hash=hashed_password,
            role=data.get("role", "student"),
            student_id=data.get("student_id", ""),
            course=data.get("course", "")
        )
        db.session.add(new_user)
        db.session.commit()

        return jsonify({"message": "User registered successfully", "user_id": new_user.id})

    except Exception as e:
        return jsonify({"error": str(e)}), 400   

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()

        user = User.query.filter_by(email=data["email"]).first()

        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        if not check_password_hash(user.password_hash, data["password"]):
            return jsonify({"error": "Invalid email or password"}), 401

        return jsonify({
            "message": "Login successful",
            "user_id": user.id,
            "name": user.name,
            "role": user.role
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/student/progress/<int:user_id>", methods=["GET"])
def progress(user_id):
    try:
        records = PredictionRecord.query.filter_by(user_id=user_id).order_by(PredictionRecord.created_at).all()

        history = [
            {
                "risk_level": r.risk_level,
                "probability_low": r.probability_low,
                "probability_medium": r.probability_medium,
                "probability_high": r.probability_high,
                "date": r.created_at.strftime("%Y-%m-%d %H:%M")
            }
            for r in records
        ]

        return jsonify({"user_id": user_id, "history": history})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        data = request.get_json()
        features = prepare_features(data)
        features_scaled = scaler.transform([features])

        prediction_num = model.predict(features_scaled)[0]
        risk_label = risk_encoder.inverse_transform([prediction_num])[0]

        shap_values = explainer.shap_values(features_scaled)
        if isinstance(shap_values, list):
            values_for_class = shap_values[prediction_num][0]
        else:
            values_for_class = shap_values[0, :, prediction_num]

        contributing_factors = [
            (name, value) for name, value in zip(FEATURE_COLUMNS, values_for_class)
            if value > 0
        ]
        contributing_factors.sort(key=lambda x: x[1], reverse=True)

        plan = {
            "immediate_actions": [],
            "short_term_goals": [],
            "resources": []
        }

        for name, value in contributing_factors[:3]:
            if name in RECOMMENDATIONS:
                plan["immediate_actions"].append(RECOMMENDATIONS[name]["immediate"])
                plan["short_term_goals"].append(RECOMMENDATIONS[name]["short_term"])
                plan["resources"].append(RECOMMENDATIONS[name]["resource"])

        if not plan["immediate_actions"]:
            plan["immediate_actions"] = ["No major risk factors detected — keep up the consistent effort!"]

        return jsonify({
            "risk_level": risk_label,
            "action_plan": plan
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_question = data.get("question", "")
        features = prepare_features(data)
        features_scaled = scaler.transform([features])

        prediction_num = model.predict(features_scaled)[0]
        risk_label = risk_encoder.inverse_transform([prediction_num])[0]

        shap_values = explainer.shap_values(features_scaled)
        if isinstance(shap_values, list):
            values_for_class = shap_values[prediction_num][0]
        else:
            values_for_class = shap_values[0, :, prediction_num]

        contributing_factors = [
            (name, value) for name, value in zip(FEATURE_COLUMNS, values_for_class)
            if value > 0
        ]
        contributing_factors.sort(key=lambda x: x[1], reverse=True)

        # NEW: build the actual recommendations, same as /recommendations does
        actions, goals, resources = [], [], []
        for name, value in contributing_factors[:3]:
            if name in RECOMMENDATIONS:
                actions.append(RECOMMENDATIONS[name]["immediate"])
                goals.append(RECOMMENDATIONS[name]["short_term"])
                resources.append(RECOMMENDATIONS[name]["resource"])

        recommendations_text = "\n".join(
            [f"- {a}" for a in actions] + [f"- {g}" for g in goals] + [f"- {r}" for r in resources]
        )
        if not recommendations_text:
            recommendations_text = "No specific concerns detected right now."

        top_factors_text = ", ".join([name for name, value in contributing_factors[:3]])

        # NEW: prompt now includes the real recommendations as context
        prompt = f"""You are EduGuide, a supportive academic assistant helping a student understand their situation.

This student's predicted academic risk level is: {risk_label}.
The top contributing factors are: {top_factors_text}.
Here are the official, pre-approved recommendations for this student:
{recommendations_text}

Answer the student's question below. If their question relates to what they should do, base your answer on the official recommendations above rather than inventing new advice. If it's a general question (like asking what a term means, or how they're feeling), answer naturally and helpfully.

Student's question: {user_question}
"""

        response = ollama.chat(model="llama3.2", messages=[
            {"role": "user", "content": prompt}
        ])

        return jsonify({
            "risk_level": risk_label,
            "answer": response["message"]["content"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/fairness-report", methods=["GET"])
def fairness_report():
    try:
        df = pd.read_csv(BASE_DIR / "data" / "processed" / "students_processed.csv")

        # gender was encoded earlier: 0 = female, 1 = male (check your encoders.pkl if unsure)
        gender_labels = encoders["gender"].classes_  # e.g. ['female', 'male']
        risk_labels = encoders["risk_level"].classes_  # e.g. ['high','low','medium']

        report = {}
        for gender_num, gender_name in enumerate(gender_labels):
            subset = df[df["gender"] == gender_num]
            total = len(subset)
            risk_counts = subset["risk_level"].value_counts(normalize=True)

            report[gender_name] = {
                risk_labels[risk_num]: round(float(pct), 3)
                for risk_num, pct in risk_counts.items()
            }

        return jsonify({
            "fairness_check": "risk distribution by gender",
            "results": report,
            "note": "Similar percentages across groups indicate the model is not biased toward a particular gender."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/teacher/students", methods=["GET"])
def teacher_students():
    try:
        students = User.query.filter_by(role="student").all()

        result = []
        for student in students:
            # Get this student's most recent prediction, if any
            latest = PredictionRecord.query.filter_by(user_id=student.id)\
                .order_by(PredictionRecord.created_at.desc()).first()

            result.append({
                "user_id": student.id,
                "name": student.name,
                "student_id": student.student_id,
                "course": student.course,
                "risk_level": latest.risk_level if latest else "No data yet",
                "last_updated": latest.created_at.strftime("%Y-%m-%d %H:%M") if latest else None
            })

        return jsonify({"students": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/teacher/student/<int:user_id>", methods=["GET"])
def teacher_student_detail(user_id):
    try:
        student = User.query.get(user_id)
        if not student or student.role != "student":
            return jsonify({"error": "Student not found"}), 404

        latest = PredictionRecord.query.filter_by(user_id=user_id)\
            .order_by(PredictionRecord.created_at.desc()).first()

        if not latest:
            return jsonify({
                "user_id": user_id,
                "name": student.name,
                "message": "No prediction data available yet for this student"
            })

        academic_data = json.loads(latest.academic_data) if latest.academic_data else {}

        # Re-run the model on their last known data, to get fresh factors + recommendations
        features = [academic_data[col] for col in FEATURE_COLUMNS]
        features_scaled = scaler.transform([features])
        prediction_num = model.predict(features_scaled)[0]

        shap_values = explainer.shap_values(features_scaled)
        if isinstance(shap_values, list):
            values_for_class = shap_values[prediction_num][0]
        else:
            values_for_class = shap_values[0, :, prediction_num]

        contributing_factors = [
            (name, value) for name, value in zip(FEATURE_COLUMNS, values_for_class)
            if value > 0
        ]
        contributing_factors.sort(key=lambda x: x[1], reverse=True)
                
        teacher_guidance = []
        for name, value in contributing_factors[:3]:
            if name in RECOMMENDATIONS:
                teacher_guidance.append({
                    "area_of_concern": name,
                    "suggested_action": RECOMMENDATIONS[name]["immediate"]
                })


             # Build a natural-language summary for the teacher, using the free local AI
        concern_list = ", ".join([f[0] for f in contributing_factors[:3]])
        guidance_list = "\n".join([
            f"- {g['area_of_concern']}: {g['suggested_action']}" for g in teacher_guidance
        ])

        teacher_prompt = f"""You are EduGuide, an assistant helping a teacher understand a student's academic risk.

Student name: {student.name}
Predicted risk level: {latest.risk_level}
Main contributing factors: {concern_list}
Relevant support actions already identified:
{guidance_list}

Write a short, clear explanation for the TEACHER (not the student) covering:
1. What happened (in plain terms, based on the factors above)
2. Why this is a concern
3. What the teacher specifically can do to help this student

Keep it brief, professional, and actionable. Do not invent facts not given above.
"""

        ai_response = ollama.chat(model="llama3.2", messages=[
            {"role": "user", "content": teacher_prompt}
        ])
        teacher_summary = ai_response["message"]["content"]

        return jsonify({
            "user_id": user_id,
            "name": student.name,
            "student_id": student.student_id,
            "course": student.course,
            "risk_level": latest.risk_level,
            "academic_data": academic_data,
            "areas_of_concern": [f[0] for f in contributing_factors[:3]],
            "teacher_guidance": teacher_guidance,
            "teacher_summary": teacher_summary
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/teacher/high-risk", methods=["GET"])
def teacher_high_risk():
    try:
        students = User.query.filter_by(role="student").all()

        high_risk_list = []
        for student in students:
            latest = PredictionRecord.query.filter_by(user_id=student.id)\
                .order_by(PredictionRecord.created_at.desc()).first()

            if latest and latest.risk_level == "high":
                high_risk_list.append({
                    "user_id": student.id,
                    "name": student.name,
                    "student_id": student.student_id,
                    "course": student.course,
                    "probability_high": latest.probability_high,
                    "last_updated": latest.created_at.strftime("%Y-%m-%d %H:%M")
                })

        # Sort so the most concerning (highest probability) students appear first
        high_risk_list.sort(key=lambda s: s["probability_high"], reverse=True)

        return jsonify({
            "count": len(high_risk_list),
            "high_risk_students": high_risk_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/student/alerts/<int:user_id>", methods=["GET"])
def student_alerts(user_id):
    try:
        records = PredictionRecord.query.filter_by(user_id=user_id)\
            .order_by(PredictionRecord.created_at.desc()).limit(2).all()

        if len(records) < 2:
            return jsonify({"alert": False, "message": "Not enough history yet to detect changes."})

        latest, previous = records[0], records[1]

        latest_severity = RISK_SEVERITY.get(latest.risk_level, 0)
        previous_severity = RISK_SEVERITY.get(previous.risk_level, 0)

        if latest_severity > previous_severity:
            return jsonify({
                "alert": True,
                "message": f"Risk increased from {previous.risk_level} to {latest.risk_level}.",
                "previous_risk": previous.risk_level,
                "current_risk": latest.risk_level
            })
        else:
            return jsonify({
                "alert": False,
                "message": "No sudden increase detected.",
                "previous_risk": previous.risk_level,
                "current_risk": latest.risk_level
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/teacher/alerts", methods=["GET"])
def teacher_alerts():
    try:
        students = User.query.filter_by(role="student").all()
        alerts = []

        for student in students:
            records = PredictionRecord.query.filter_by(user_id=student.id)\
                .order_by(PredictionRecord.created_at.desc()).limit(2).all()

            if len(records) < 2:
                continue  # skip students without enough history

            latest, previous = records[0], records[1]
            latest_severity = RISK_SEVERITY.get(latest.risk_level, 0)
            previous_severity = RISK_SEVERITY.get(previous.risk_level, 0)

            if latest_severity > previous_severity:
                alerts.append({
                    "user_id": student.id,
                    "name": student.name,
                    "previous_risk": previous.risk_level,
                    "current_risk": latest.risk_level,
                    "date": latest.created_at.strftime("%Y-%m-%d %H:%M")
                })

        return jsonify({"count": len(alerts), "alerts": alerts})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/simulate", methods=["POST"])
def simulate():
    try:
        data = request.get_json()

        validation_error = validate_features(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        features = prepare_features(data)
        features_scaled = scaler.transform([features])

        prediction_num = model.predict(features_scaled)[0]
        probabilities = probability_model.predict_proba(features_scaled)[0]
        risk_label = risk_encoder.inverse_transform([prediction_num])[0]

        prob_dict = {
            str(risk_encoder.inverse_transform([int(class_id)])[0]).lower(): round(float(p), 6)
            for class_id, p in zip(probability_model.classes_, probabilities)
        }

        return jsonify({"risk_level": risk_label, "probabilities": prob_dict})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

from collections import defaultdict

@app.route("/admin/overview", methods=["GET"])
def admin_overview():
    try:
        return jsonify({
            "total_students": User.query.filter_by(role="student").count(),
            "total_teachers": User.query.filter_by(role="teacher").count(),
            "total_predictions": PredictionRecord.query.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/users", methods=["GET"])
def admin_users():
    try:
        users = User.query.order_by(User.created_at.desc()).all()
        result = []
        for u in users:
            result.append({
                "user_id": u.id,
                "name": u.name,
                "email": u.email,
                "role": u.role,
                "student_id": u.student_id,
                "course": u.course,
                "created_at": u.created_at.strftime("%Y-%m-%d %H:%M"),
                "warning_count": Warning.query.filter_by(user_id=u.id).count()
            })
        return jsonify({"users": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/users/<int:user_id>", methods=["DELETE"])
def admin_delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        if user.role == "admin":
            return jsonify({"error": "Admin accounts cannot be deleted from this panel."}), 403

        PredictionRecord.query.filter_by(user_id=user_id).delete()
        db.session.delete(user)
        db.session.commit()
        return jsonify({"message": "User deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/predictions-timeline", methods=["GET"])
def admin_predictions_timeline():
    try:
        records = PredictionRecord.query.order_by(PredictionRecord.created_at).all()
        daily_counts = defaultdict(int)
        for r in records:
            daily_counts[r.created_at.strftime("%Y-%m-%d")] += 1

        timeline = [{"date": day, "count": count} for day, count in sorted(daily_counts.items())]
        return jsonify({"timeline": timeline, "total": len(records)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/system-status", methods=["GET"])
def admin_system_status():
    status = {}
    try:
        User.query.first()
        status["database"] = "online"
    except Exception:
        status["database"] = "offline"

    try:
        _ = model.predict
        status["ml_model"] = "online"
    except Exception:
        status["ml_model"] = "offline"

    try:
        ollama.list()
        status["chatbot"] = "online"
    except Exception:
        status["chatbot"] = "offline"

    return jsonify(status)

@app.route("/admin/activity-feed", methods=["GET"])
def admin_activity_feed():
    try:
        events = []

        for u in User.query.order_by(User.created_at.desc()).limit(10).all():
            events.append({"type": "registration", "text": f"{u.name} registered as {u.role}", "timestamp": u.created_at})

        for p in PredictionRecord.query.order_by(PredictionRecord.created_at.desc()).limit(10).all():
            student = User.query.get(p.user_id)
            name = student.name if student else "A student"
            events.append({"type": "prediction", "text": f"{name} completed an assessment — {p.risk_level} risk", "timestamp": p.created_at})

        events.sort(key=lambda e: e["timestamp"], reverse=True)
        feed = [{"type": e["type"], "text": e["text"], "time": e["timestamp"].strftime("%Y-%m-%d %H:%M")} for e in events[:15]]
        return jsonify({"feed": feed})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    try:
        data = request.get_json()
        user = User.query.get(data.get("user_id"))
        if user:
            user.last_seen = datetime.utcnow()
            db.session.commit()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/active-users", methods=["GET"])
def admin_active_users():
    try:
        cutoff = datetime.utcnow() - timedelta(seconds=60)
        active = User.query.filter(User.last_seen != None, User.last_seen >= cutoff).all()
        return jsonify({
            "count": len(active),
            "active_users": [{"user_id": u.id, "name": u.name, "role": u.role} for u in active]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/users/<int:user_id>/warn", methods=["POST"])
def admin_warn_user(user_id):
    try:
        data = request.get_json()
        reason = (data.get("reason") or "").strip()
        if not reason:
            return jsonify({"error": "A reason is required."}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        warning = Warning(user_id=user_id, reason=reason)
        db.session.add(warning)
        db.session.commit()
        return jsonify({"message": "Warning issued", "warning_id": warning.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/admin/user/<int:user_id>", methods=["GET"])
def admin_user_detail(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        base = {
            "user_id": user.id, "name": user.name, "email": user.email,
            "role": user.role, "student_id": user.student_id, "course": user.course,
            "joined": user.created_at.strftime("%Y-%m-%d %H:%M")
        }

        if user.role == "student":
            latest = PredictionRecord.query.filter_by(user_id=user_id)\
                .order_by(PredictionRecord.created_at.desc()).first()
            if latest:
                base["risk_level"] = latest.risk_level
                base["academic_data"] = json.loads(latest.academic_data) if latest.academic_data else {}
            else:
                base["risk_level"] = None

        warnings = Warning.query.filter_by(user_id=user_id).order_by(Warning.created_at.desc()).all()
        base["warnings"] = [{"reason": w.reason, "date": w.created_at.strftime("%Y-%m-%d %H:%M")} for w in warnings]

        return jsonify(base)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


def get_chatroom_settings():
    settings = ChatroomSettings.query.first()
    if not settings:
        settings = ChatroomSettings(locked=False)
        db.session.add(settings)
        db.session.commit()
    return settings
 
 
# =====================================================
# ROUTES
# =====================================================
 
@app.route("/teacher/remarks", methods=["POST"])
def send_remark():
    data = request.get_json()
    student_user_id = data.get("student_user_id")
    message = data.get("message")
 
    if not student_user_id or not message:
        return jsonify({"error": "student_user_id and message are required"}), 400
 
    remark = Remark(student_user_id=student_user_id, message=message)
    db.session.add(remark)
    db.session.commit()
 
    return jsonify({"success": True, "id": remark.id})
 
 
@app.route("/student/remarks/<int:user_id>", methods=["GET"])
def get_remarks(user_id):
    remarks = (
        Remark.query
        .filter_by(student_user_id=user_id)
        .order_by(Remark.created_at.desc())
        .all()
    )
 
    return jsonify({
        "remarks": [
            {
                "id": r.id,
                "message": r.message,
                "created_at": r.created_at.strftime("%b %d, %Y %I:%M %p"),
                "read": r.read
            }
            for r in remarks
        ]
    })
 
 
@app.route("/student/remarks/<int:remark_id>/read", methods=["POST"])
def mark_remark_read(remark_id):
    remark = Remark.query.get(remark_id)
    if not remark:
        return jsonify({"error": "Remark not found"}), 404
 
    remark.read = True
    db.session.commit()
    return jsonify({"success": True})
 
 
@app.route("/chatroom/messages", methods=["GET"])
def get_chatroom_messages():
    messages = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    settings = get_chatroom_settings()
 
    return jsonify({
        "messages": [
            {
                "user_id": m.user_id,
                "name": m.name,
                "role": m.role,
                "message": m.message,
                "timestamp": m.timestamp.strftime("%b %d, %I:%M %p")
            }
            for m in messages
        ],
        "locked": settings.locked
    })
 
 
@app.route("/chatroom/messages", methods=["POST"])
def post_chatroom_message():
    data = request.get_json()
    role = data.get("role")
 
    settings = get_chatroom_settings()
    if settings.locked and role != "teacher":
        return jsonify({"error": "The chatroom is currently locked by your teacher."}), 403
 
    msg = ChatMessage(
        user_id=data.get("user_id"),
        name=data.get("name", "Unknown"),
        role=role,
        message=data.get("message", "")
    )
    db.session.add(msg)
    db.session.commit()
 
    return jsonify({"success": True})
 
 
@app.route("/chatroom/lock", methods=["POST"])
def set_chatroom_lock():
    data = request.get_json()
    settings = get_chatroom_settings()
    settings.locked = bool(data.get("locked", False))
    db.session.commit()
 
    return jsonify({"success": True, "locked": settings.locked})

@app.route("/leaderboard/<int:user_id>", methods=["GET"])
def leaderboard(user_id):
    try:
        students = User.query.filter_by(role="student").all()
        rows = []
        compared_ids = set()

        for student in students:
            records = (
                PredictionRecord.query
                .filter_by(user_id=student.id)
                .order_by(PredictionRecord.created_at.desc())
                .limit(2)
                .all()
            )

            # Need two check-ins to compare "last" with "current"
            if len(records) < 2:
                continue

            compared_ids.add(student.id)
            latest, previous = records[0], records[1]

            new_data = json.loads(latest.academic_data) if latest.academic_data else {}
            old_data = json.loads(previous.academic_data) if previous.academic_data else {}

            score_gain = max(0, new_data.get("avg_score", 0) - old_data.get("avg_score", 0))
            attendance_gain = max(0, new_data.get("attendance", 0) - old_data.get("attendance", 0))
            study_gain = max(0, new_data.get("hours_studied", 0) - old_data.get("hours_studied", 0))

            # Points only for improvement, each part has a maximum
            points = min(40, round(score_gain * 2))
            points += min(30, round(attendance_gain))
            points += min(15, round(study_gain * 5))

            # Bonus when the risk level got better
            latest_level = RISK_SEVERITY.get(latest.risk_level, 0)
            previous_level = RISK_SEVERITY.get(previous.risk_level, 0)
            if latest_level < previous_level:
                points += 30

            if points <= 0:
                continue

            parts = (student.name or "").split()
            if len(parts) > 1:
                display_name = f"{parts[0]} {parts[-1][0]}."
                initials = (parts[0][0] + parts[-1][0]).upper()
            elif parts:
                display_name = parts[0]
                initials = parts[0][0].upper()
            else:
                display_name = "Student"
                initials = "S"

            rows.append({
                "user_id": student.id,
                "name": display_name,
                "initials": initials,
                "points": points
            })

        rows.sort(key=lambda row: (-row["points"], row["name"]))

        # Students with equal points share the same rank
        for index, row in enumerate(rows):
            if index > 0 and row["points"] == rows[index - 1]["points"]:
                row["rank"] = rows[index - 1]["rank"]
            else:
                row["rank"] = index + 1

        top = rows[:5]
        top_list = [
            {
                "rank": row["rank"],
                "name": row["name"],
                "initials": row["initials"],
                "points": row["points"],
                "is_me": row["user_id"] == user_id
            }
            for row in top
        ]

        mine = next((row for row in rows if row["user_id"] == user_id), None)
        my_in_top = any(row["user_id"] == user_id for row in top)

        if mine:
            status = "ranked"
        elif user_id in compared_ids:
            status = "no_improvement"
        else:
            status = "needs_second_checkin"

        my_points = mine["points"] if mine else 0

        points_to_top5 = None
        if status != "needs_second_checkin" and not my_in_top:
            fifth_points = rows[4]["points"] if len(rows) >= 5 else 0
            points_to_top5 = max(1, fifth_points + 1 - my_points)

        return jsonify({
            "top": top_list,
            "me": {
                "status": status,
                "rank": mine["rank"] if mine else None,
                "points": my_points,
                "points_to_top5": points_to_top5
            },
            "total_ranked": len(rows)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/student/profile/<int:user_id>", methods=["GET"])
def student_profile(user_id):
    try:
        user = User.query.get(user_id)
        if not user or user.role != "student":
            return jsonify({"error": "Student not found"}), 404

        return jsonify({
            "name": user.name,
            "student_id": user.student_id or ""
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400                                                                                                                         

if __name__ == "__main__":
    app.run(debug=True, port=5000)

