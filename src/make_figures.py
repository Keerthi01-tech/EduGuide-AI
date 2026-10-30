import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("matplotlib is not installed.")
    print("Type this, press Enter, then run this script again:")
    print("pip install matplotlib")
    raise SystemExit

import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_curve, auc
)

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    XGBClassifier = None

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "students_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
OUT_DIR = BASE_DIR.parent / "EduGuide_paper_figures"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["risk_level"])
y = df["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
y_test_arr = np.array(y_test)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000, C=1.0, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        random_state=42, n_estimators=200),
}
if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        random_state=42, eval_metric="mlogloss")

colors = {
    "Logistic Regression": "#c0272d",
    "Decision Tree": "#b87a1f",
    "Random Forest": "#6b48b8",
    "XGBoost": "#2c6fb7",
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    results[name] = {
        "pred": pred,
        "proba": model.predict_proba(X_test_s),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test, pred, labels=[0], average="macro", zero_division=0),
        "recall": recall_score(
            y_test, pred, labels=[0], average="macro", zero_division=0),
        "f1": f1_score(
            y_test, pred, labels=[0], average="macro", zero_division=0),
    }

print("Values drawn (high-risk class for precision, recall, F1):")
for name, r in results.items():
    print(f"  {name}: accuracy {r['accuracy']:.4f}, "
          f"precision {r['precision']:.4f}, recall {r['recall']:.4f}, "
          f"F1 {r['f1']:.4f}")


def bar_chart(key, ylabel, title, ylim, filename):
    names = list(results.keys())
    values = [results[n][key] for n in names]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    bars = ax.bar(range(len(names)), values,
                  color=[colors[n] for n in names], width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v * 100:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT_DIR / filename)
    plt.close(fig)


bar_chart("accuracy", "Accuracy", "Accuracy of the four algorithms",
          (0.90, 1.0), "fig_accuracy.png")
bar_chart("precision", "Precision (high risk)",
          "High-risk precision of the four algorithms",
          (0.70, 1.0), "fig_precision.png")
bar_chart("recall", "Recall (high risk)",
          "High-risk recall of the four algorithms",
          (0.70, 1.0), "fig_recall.png")
bar_chart("f1", "F1-score (high risk)",
          "High-risk F1-score of the four algorithms",
          (0.70, 1.0), "fig_f1.png")

# ROC curves: high-risk class against all other classes
fig, ax = plt.subplots(figsize=(3.5, 3.2))
is_high = (y_test_arr == 0).astype(int)
for name, r in results.items():
    fpr, tpr, _ = roc_curve(is_high, r["proba"][:, 0])
    ax.plot(fpr, tpr, color=colors[name], lw=1.5,
            label=f"{name} (AUC {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves, high-risk class (test set)",
             fontsize=9, fontweight="bold")
ax.legend(loc="lower right", fontsize=7)
ax.grid(alpha=0.3)
fig.savefig(OUT_DIR / "fig_roc_high_risk.png")
plt.close(fig)

# ROC curves of Logistic Regression for each of the three classes
lr_proba = results["Logistic Regression"]["proba"]
fig, ax = plt.subplots(figsize=(3.5, 3.2))
for cls, label, col in [(0, "High", "#dc2626"),
                        (1, "Low", "#16a34a"),
                        (2, "Medium", "#d97706")]:
    fpr, tpr, _ = roc_curve((y_test_arr == cls).astype(int),
                            lr_proba[:, cls])
    ax.plot(fpr, tpr, color=col, lw=1.5,
            label=f"{label} risk (AUC {auc(fpr, tpr):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves, Logistic Regression (test set)",
             fontsize=9, fontweight="bold")
ax.legend(loc="lower right", fontsize=7)
ax.grid(alpha=0.3)
fig.savefig(OUT_DIR / "fig_roc_logistic_classes.png")
plt.close(fig)

# Confusion matrix of Logistic Regression
lr_pred = results["Logistic Regression"]["pred"]
cm = confusion_matrix(y_test, lr_pred, labels=[0, 1, 2])
cm_pct = cm / cm.sum(axis=1, keepdims=True)
print("\nConfusion matrix, Logistic Regression "
      "(rows true, columns predicted; high, low, medium):")
print(cm)
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
tick_labels = ["High", "Low", "Medium"]
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(tick_labels)
ax.set_yticklabels(tick_labels)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion matrix, Logistic Regression\n"
             "(test set, 1,700 students)", fontsize=9, fontweight="bold")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j] * 100:.1f}%)",
                ha="center", va="center", fontsize=8,
                color="white" if cm_pct[i, j] > 0.5 else "black")
ax.spines[:].set_visible(True)
fig.savefig(OUT_DIR / "fig_confusion_logistic.png")
plt.close(fig)

# Class distribution of the whole dataset
counts = y.value_counts()
order = [(1, "Low risk", "#16a34a"),
         (2, "Medium risk", "#d97706"),
         (0, "High risk", "#dc2626")]
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for i, (code, label, col) in enumerate(order):
    n = int(counts[code])
    ax.bar(i, n, color=col, width=0.6)
    ax.text(i, n + 100, f"{n:,}\n({n / len(y) * 100:.1f}%)",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(3))
ax.set_xticklabels([o[1] for o in order])
ax.set_ylabel("Students")
ax.set_ylim(0, counts.max() * 1.25)
ax.set_title(f"Risk-level distribution (n = {len(y):,})",
             fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.savefig(OUT_DIR / "fig_class_distribution.png")
plt.close(fig)

# SHAP importance of the saved Logistic Regression model
saved_model = joblib.load(MODEL_PATH)
saved_scaler = joblib.load(SCALER_PATH)
X_all_scaled = saved_scaler.transform(X)
explainer = shap.LinearExplainer(saved_model, X_all_scaled)
shap_values = explainer.shap_values(X_all_scaled)
if isinstance(shap_values, list):
    importance = np.mean(
        [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
else:
    arr = np.abs(shap_values)
    if arr.ndim == 3:
        importance = arr.mean(axis=(0, 2))
    else:
        importance = arr.mean(axis=0)
imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)

top = imp.head(10)
fig, ax = plt.subplots(figsize=(3.5, 3.0))
ax.barh(list(top.index[::-1]), list(top.values[::-1]), color="#4f46e5")
ax.set_xlabel("Mean absolute SHAP value")
ax.set_title("Top 10 factors (Logistic Regression)",
             fontsize=9, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_axisbelow(True)
fig.savefig(OUT_DIR / "fig_shap_top10.png")
plt.close(fig)

groups = {
    "Attendance and\nabsence": ["absence_days", "attendance", "Absent"],
    "Scores and\ngrades": ["GPA", "exam_score", "previous_scores",
                           "avg_score", "C_score", "Python_score",
                           "Java_score", "Grade"],
    "Study\nhabits": ["hours_studied", "study_time"],
}
used = [c for cols in groups.values() for c in cols]
groups["All other\nfactors"] = [c for c in X.columns if c not in used]
total = imp.sum()
group_share = {g: imp[cols].sum() / total * 100
               for g, cols in groups.items()}
print("\nSHAP importance share by group (percent of total):")
for g, v in group_share.items():
    print(f"  {g.replace(chr(10), ' ')}: {v:.1f}%")

fig, ax = plt.subplots(figsize=(3.5, 2.6))
vals = list(group_share.values())
bars = ax.bar(range(len(vals)), vals,
              color=["#dc2626", "#4f46e5", "#16a34a", "#9ca3af"],
              width=0.6)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(list(group_share.keys()), fontsize=8)
ax.set_ylabel("Share of total importance (%)")
ax.set_ylim(0, max(vals) * 1.2)
ax.set_title("SHAP importance by factor group",
             fontsize=9, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
fig.savefig(OUT_DIR / "fig_shap_groups.png")
plt.close(fig)

print("\nDone. Images saved in:", OUT_DIR)
for f in sorted(OUT_DIR.glob("*.png")):
    print("  ", f.name)