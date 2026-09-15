import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "students.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "students_processed.csv"
ENCODERS_PATH = BASE_DIR / "models" / "encoders.pkl"

df = pd.read_csv(RAW_DATA_PATH)
print("Loaded data with shape:", df.shape)

categorical_cols = [
    "gender", "Grade", "absence_days", "Course", "test_prep",
    "motivation_level", "internet_access", "risk_level"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"Encoded '{col}': classes -> {list(le.classes_)}")

if "Student_id" in df.columns:
    df = df.drop(columns=["Student_id"])

PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(PROCESSED_DATA_PATH, index=False)
print("Processed data saved to:", PROCESSED_DATA_PATH)

ENCODERS_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(encoders, ENCODERS_PATH)
print("Encoders saved to:", ENCODERS_PATH)