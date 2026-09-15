import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "students_processed.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(PROCESSED_DATA_PATH)

X = df.drop(columns=["risk_level"])
X_scaled = scaler.transform(X)

explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer.shap_values(X_scaled)

# shap_values is a list: one array per class (high/low/medium)
# Average the importance across all 3 classes, then print as plain text ranking
# Handle both possible SHAP output formats safely
# Handle whichever SHAP output shape/format we got
if isinstance(shap_values, list):
    # Case: list of arrays, one per class
    mean_abs_per_class = [np.abs(sv).mean(axis=0) for sv in shap_values]
    overall_importance = np.mean(mean_abs_per_class, axis=0)
else:
    shap_array = np.abs(shap_values)
    if shap_array.ndim == 3:
        # Shape: (samples, features, classes) -> average over samples AND classes
        overall_importance = shap_array.mean(axis=(0, 2))
    else:
        # Shape: (samples, features) -> average over samples only
        overall_importance = shap_array.mean(axis=0)

print("Shape check - overall_importance should be 1D with length =", len(X.columns))
print("Actual shape:", overall_importance.shape)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": overall_importance
}).sort_values("importance", ascending=False)

print("\nFeature importance ranking (text, no guessing needed):")
print(importance_df.to_string(index=False))