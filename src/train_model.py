import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = (
    BASE_DIR / "data" / "processed" / "students_processed.csv"
)
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
PROBABILITY_MODEL_PATH = BASE_DIR / "models" / "probability_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

df = pd.read_csv(PROCESSED_DATA_PATH)
X = df.drop(columns=["risk_level"])
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        C=0.001,
        random_state=42
    ),
    "DecisionTree": DecisionTreeClassifier(
        random_state=42
    ),
    "RandomForest": RandomForestClassifier(
        random_state=42,
        n_estimators=200
    )
}

if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss"
    )

best_model_name = None
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\n----- {name} -----")
    print("Accuracy:", round(accuracy, 4))
    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    if name == "LogisticRegression":
        best_score = accuracy
        best_model_name = name
        best_model = model

print(
    f"\nBest model: {best_model_name} "
    f"with accuracy {round(best_score, 4)}"
)

# The raw model is used for the risk category and SHAP explanations.
# The calibrated model is used for better probability percentages.
probability_model = CalibratedClassifierCV(
    estimator=LogisticRegression(
        max_iter=5000,
        C=0.001,
        random_state=42
    ),
    method="sigmoid",
    cv=5
)
probability_model.fit(X_train_scaled, y_train)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(best_model, MODEL_PATH)
joblib.dump(probability_model, PROBABILITY_MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("Model saved to:", MODEL_PATH)
print(
    "Calibrated probability model saved to:",
    PROBABILITY_MODEL_PATH
)
print("Scaler saved to:", SCALER_PATH)