import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, make_scorer
)

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "students_processed.csv"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["risk_level"])
y = df["risk_level"]


def make_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=5000, C=1.0, random_state=42
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(
            random_state=42, n_estimators=200
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            random_state=42, eval_metric="mlogloss"
        )
    return models


# ---------- Part 1: test-set results (same split as training) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_test_arr = np.array(y_test)

print("=== PART 1: TEST SET (1,700 students) ===")
print("class 0 = high risk, 1 = low risk, 2 = medium risk")
lr_pred = None
lr_proba = None
for name, model in make_models().items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)
    if name == "LogisticRegression":
        lr_pred = pred
        lr_proba = proba
    print(f"\n{name}")
    print("  accuracy        :", round(accuracy_score(y_test, pred), 4))
    print("  high precision  :", round(precision_score(
        y_test, pred, labels=[0], average="macro", zero_division=0), 4))
    print("  high recall     :", round(recall_score(
        y_test, pred, labels=[0], average="macro", zero_division=0), 4))
    print("  high F1         :", round(f1_score(
        y_test, pred, labels=[0], average="macro", zero_division=0), 4))
    print("  macro F1        :", round(
        f1_score(y_test, pred, average="macro"), 4))
    print("  ROC-AUC (macro) :", round(roc_auc_score(
        y_test, proba, multi_class="ovr", average="macro"), 4))
    print("  ROC-AUC (high)  :", round(roc_auc_score(
        (y_test_arr == 0).astype(int), proba[:, 0]), 4))

# ---------- Part 2: 5-fold cross-validation on all 8,500 rows ----------
print("\n=== PART 2: 5-FOLD CROSS-VALIDATION (all 8,500 rows) ===")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    "accuracy": "accuracy",
    "high_recall": make_scorer(
        recall_score, labels=[0], average="macro", zero_division=0
    ),
    "auc_ovr": "roc_auc_ovr",
}
for name, model in make_models().items():
    pipe = make_pipeline(StandardScaler(), model)
    res = cross_validate(pipe, X, y, cv=skf, scoring=scoring)
    print(f"\n{name}")
    for key in ["accuracy", "high_recall", "auc_ovr"]:
        vals = res["test_" + key]
        print(f"  {key:12s}: mean {vals.mean():.4f}  "
              f"std {vals.std():.4f}")

# ---------- Part 3: Logistic Regression by gender (test set) ----------
print("\n=== PART 3: LOGISTIC REGRESSION BY GENDER (test set) ===")
print("gender encoding: 0 = female, 1 = male")
gender = np.array(X_test["gender"])
for code, label in [(0, "female"), (1, "male")]:
    mask = gender == code
    n = int(mask.sum())
    n_high = int(((y_test_arr == 0) & mask).sum())
    caught = int(((lr_pred == 0) & (y_test_arr == 0) & mask).sum())
    n_notHigh = int(((y_test_arr != 0) & mask).sum())
    false_high = int(((lr_pred == 0) & (y_test_arr != 0) & mask).sum())
    acc = float((lr_pred[mask] == y_test_arr[mask]).mean())
    print(f"\n{label}: students {n}, truly high-risk {n_high}")
    print(f"  high-risk caught : {caught} of {n_high} "
          f"(recall {caught / n_high:.4f})")
    print(f"  wrongly flagged high: {false_high} of {n_notHigh} "
          f"(rate {false_high / n_notHigh:.4f})")
    print(f"  accuracy         : {acc:.4f}")

# ---------- Part 4: Brier score (lower is better) ----------
print("\n=== PART 4: BRIER SCORE, test set (lower is better) ===")
cal = CalibratedClassifierCV(
    estimator=LogisticRegression(max_iter=5000, C=1.0, random_state=42),
    method="sigmoid", cv=5
)
cal.fit(X_train_s, y_train)
cal_proba = cal.predict_proba(X_test_s)
onehot = np.eye(3)[y_test_arr]


def brier(p):
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


print("plain Logistic Regression     :", round(brier(lr_proba), 4))
print("calibrated Logistic Regression:", round(brier(cal_proba), 4))