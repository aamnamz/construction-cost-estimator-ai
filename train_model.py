import pandas as pd
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
df = pd.read_csv("building_project_cost_dataset_FIXED.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ─────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────
df.drop(columns=["project_id", "cost_category"], inplace=True)

cat_cols = ["project_type", "project_location", "construction_type", "foundation_type"]
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["total_project_cost"])
y = df["total_project_cost"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────
# 3. RANDOM FOREST
# ─────────────────────────────────────────
print("\n--- Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mae  = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2   = r2_score(y_test, rf_preds)
print(f"MAE  : {rf_mae:,.0f}")
print(f"RMSE : {rf_rmse:,.0f}")
print(f"R²   : {rf_r2:.4f}")

# ─────────────────────────────────────────
# 4. XGBOOST
# ─────────────────────────────────────────
print("\n--- XGBoost ---")
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1,
                   random_state=42, n_jobs=-1, verbosity=0)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

xgb_mae  = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2   = r2_score(y_test, xgb_preds)
print(f"MAE  : {xgb_mae:,.0f}")
print(f"RMSE : {xgb_rmse:,.0f}")
print(f"R²   : {xgb_r2:.4f}")

# ─────────────────────────────────────────
# 5. LINEAR REGRESSION  (3rd model)
# ─────────────────────────────────────────
print("\n--- Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_mae  = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2   = r2_score(y_test, lr_preds)
print(f"MAE  : {lr_mae:,.0f}")
print(f"RMSE : {lr_rmse:,.0f}")
print(f"R²   : {lr_r2:.4f}")

# ─────────────────────────────────────────
# 6. MODEL COMPARISON TABLE
# ─────────────────────────────────────────
print("\n=== MODEL COMPARISON ===")
comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost", "Linear Regression"],
    "MAE":   [rf_mae, xgb_mae, lr_mae],
    "RMSE":  [rf_rmse, xgb_rmse, lr_rmse],
    "R²":    [rf_r2, xgb_r2, lr_r2],
})
print(comparison.to_string(index=False))

# ─────────────────────────────────────────
# 7. PICK BEST MODEL (by R²)
# ─────────────────────────────────────────
scores = {"Random Forest": rf_r2, "XGBoost": xgb_r2, "Linear Regression": lr_r2}
best_name  = max(scores, key=scores.get)
best_model = {"Random Forest": rf, "XGBoost": xgb, "Linear Regression": lr}[best_name]

print(f"\nBest model: {best_name} (R² = {scores[best_name]:.4f})")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Save comparison metrics for the app to load
comparison.to_csv("model_comparison.csv", index=False)
print("Model saved as best_model.pkl | Comparison saved as model_comparison.csv")

# ─────────────────────────────────────────
# 8. SHAP EXPLAINABILITY
# ─────────────────────────────────────────
print("\n--- SHAP Analysis ---")

explainer   = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()
print("SHAP summary plot saved as shap_summary.png")

shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=150)
plt.close()
print("SHAP importance plot saved as shap_importance.png")

# ─────────────────────────────────────────
# 9. LIME EXPLAINABILITY
# ─────────────────────────────────────────
print("\n--- LIME Analysis ---")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data  = X_train.values,
    feature_names  = X.columns.tolist(),
    mode           = "regression",
    random_state   = 42,
)

# Explain a sample instance
sample_instance = X_test.values[0]
lime_exp = lime_explainer.explain_instance(
    data_row       = sample_instance,
    predict_fn     = best_model.predict,
    num_features   = 10,
)

# Save LIME explanation as PNG
fig_lime = lime_exp.as_pyplot_figure()
fig_lime.tight_layout()
fig_lime.savefig("lime_explanation.png", dpi=150, bbox_inches="tight")
plt.close(fig_lime)
print("LIME explanation saved as lime_explanation.png")

print("LIME explainer is rebuilt by app.py at runtime.")

# ─────────────────────────────────────────
# 10. SAMPLE PREDICTION
# ─────────────────────────────────────────
print("\n--- Sample Prediction ---")
sample     = X_test.iloc[[0]]
prediction = best_model.predict(sample)[0]
actual     = y_test.iloc[0]
print(f"Predicted cost : {prediction:,.0f}")
print(f"Actual cost    : {actual:,.0f}")
print(f"Difference     : {abs(prediction - actual):,.0f}")

print("\nDone! Next step: run app.py for the Streamlit UI.")
