from app import app, db
from models_db import User
from werkzeug.security import generate_password_hash

with app.app_context():
    # Force SQLAlchemy to drop old tables and create fresh ones with new columns
    db.drop_all()
    db.create_all()

    print("=== Create Admin Account ===")
    name = input("Admin name: ").strip()
    email = input("Admin email: ").strip()
    password = input("Admin password: ").strip()

    existing = User.query.filter_by(email=email).first()
    if existing:
        print(f"A user with this email already exists (user_id {existing.id}). Aborting.")
    else:
        admin = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role="admin"
        )
        db.session.add(admin)
        db.session.commit()
        print(f"Admin account created successfully. user_id = {admin.id}")