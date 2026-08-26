import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import os

st.set_page_config(page_title="AI Construction Cost Estimator", page_icon="🏗️", layout="wide")

@st.cache_resource
def load_model():
    model    = joblib.load("best_model.pkl")
    features = joblib.load("feature_names.pkl")
    return model, features

@st.cache_resource
def load_lime_explainer(features):
    if not os.path.exists("building_project_cost_dataset_FIXED.csv"):
        return None

    df = pd.read_csv("building_project_cost_dataset_FIXED.csv")
    df.drop(columns=["project_id", "cost_category", "total_project_cost"], inplace=True)
    for col, mapping in {
        "project_type": {"Commercial": 0, "Industrial": 1, "Residential": 2},
        "project_location": {"Rural": 0, "Semi-Urban": 1, "Urban": 2},
        "construction_type": {"Hybrid": 0, "RCC": 1, "Steel": 2},
        "foundation_type": {"Pile": 0, "Raft": 1, "Strip": 2},
    }.items():
        df[col] = df[col].map(mapping)

    df = df[features].apply(pd.to_numeric)
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=df.values,
        feature_names=features,
        mode="regression",
        random_state=42,
    )

@st.cache_data
def load_comparison():
    if os.path.exists("model_comparison.csv"):
        return pd.read_csv("model_comparison.csv")
    return None

@st.cache_data
def load_minimum_cost_per_sqft():
    if os.path.exists("building_project_cost_dataset_FIXED.csv"):
        df = pd.read_csv(
            "building_project_cost_dataset_FIXED.csv",
            usecols=["total_area_sqft", "total_project_cost"],
        )
        cost_per_sqft = df["total_project_cost"] / df["total_area_sqft"]
        return float(cost_per_sqft.min())
    return 0.0

model, feature_names = load_model()
comparison_df  = load_comparison()
minimum_cost_per_sqft = load_minimum_cost_per_sqft()

# ── Encoding maps (must match LabelEncoder order = alphabetical) ──
project_type_map = {
    "Residential (House / Apartment)":  2,
    "Commercial (Shop / Office)":       0,
    "Industrial (Factory / Warehouse)": 1,
}
project_location_map = {
    "Urban (City Centre)":        2,
    "Semi-Urban (Suburban Area)": 1,
    "Rural (Village / Outskirts)":0,
}
construction_type_map = {
    "RCC (Concrete)":            1,
    "Steel Structure":           2,
    "Hybrid (Concrete + Steel)": 0,
}
foundation_type_map = {
    "Strip Foundation (standard houses)":            2,
    "Raft Foundation (soft soil / large buildings)": 1,
    "Pile Foundation (high-rise / weak soil)":       0,
}

MATERIAL_QUALITY = {
    "Basic (low-cost materials)":     0.4,
    "Standard (mid-range materials)": 0.7,
    "Premium (high-end materials)":   0.95,
}
LABOR_QUALITY = {
    "Unskilled / cheap labor":            0.8,
    "Semi-skilled labor":                 1.3,
    "Highly skilled / experienced labor": 2.0,
}
MATERIAL_COST = {
    "Low market prices":     0.8,
    "Average market prices": 1.5,
    "High market prices":    2.2,
}
INFLATION = {
    "Low (stable economy)":   3.0,
    "Moderate":               7.0,
    "High (rising prices)":  14.0,
}
EXPERIENCE = {
    "Less than 2 years": 1,
    "2–5 years":         4,
    "5–15 years":       10,
    "15+ years (expert)":20,
}
TEMPERATURE = {
    "Cold (below 15°C)":   10.0,
    "Moderate (15–30°C)":  25.0,
    "Hot (above 30°C)":    38.0,
}
RAINFALL = {
    "Dry area (little rain)": 0.1,
    "Moderate rainfall":      0.4,
    "Heavy rainfall area":    0.8,
}
WEATHER_DELAY = {
    "No delays expected":          0,
    "Minor delays (1–2 weeks)":   10,
    "Moderate delays (1 month)":  30,
    "Severe delays (2+ months)":  60,
}
COMPLEXITY = {
    "Simple / standard design":       0.3,
    "Moderate complexity":            0.6,
    "Highly complex / custom design": 0.9,
}
RISK = {
    "Low risk project":                      0.2,
    "Medium risk":                           0.5,
    "High risk (difficult site/conditions)": 0.8,
}
RESOURCE_USE = {
    "Poor (lots of waste)":      0.5,
    "Average":                   0.75,
    "Efficient (minimal waste)": 0.95,
}

lime_explainer = load_lime_explainer(feature_names)

FRIENDLY_NAMES = {
    "total_area_sqft"            : "Total Area",
    "num_floors"                 : "Number of Floors",
    "material_quality_index"     : "Material Quality",
    "labor_cost_index"           : "Labor Cost",
    "material_cost_index"        : "Material Cost",
    "inflation_rate"             : "Inflation Rate",
    "equipment_cost"             : "Equipment Cost",
    "contractor_experience_years": "Contractor Experience",
    "avg_temperature"            : "Temperature",
    "rainfall_index"             : "Rainfall",
    "weather_delay_days"         : "Weather Delays",
    "complexity_score"           : "Project Complexity",
    "risk_index"                 : "Risk Level",
    "resource_utilization_rate"  : "Resource Utilization",
    "project_type"               : "Project Type",
    "project_location"           : "Location",
    "construction_type"          : "Construction Type",
    "foundation_type"            : "Foundation Type",
}


def get_ai_suggestions(prompt, api_key):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type" : "application/json",
        },
        json={
            "model"     : "openai/gpt-oss-20b",
            "messages"  : [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
        },
        timeout=30
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ────────────────────────────────────────────
# UI
# ────────────────────────────────────────────
st.title("🏗️ AI Construction Cost Estimator")
st.markdown("Answer a few simple questions about your project and we'll estimate the total cost.")

# ── Model Comparison Table (always visible) ──
if comparison_df is not None:
    with st.expander("📊 Model Comparison — All 3 Trained Models", expanded=False):
        st.caption("Evaluation metrics for all predictive models (lower MAE/RMSE = better, higher R² = better)")
        r2_col = next((col for col in comparison_df.columns if col.startswith("R") and col != "RMSE"), None)
        best_row = comparison_df.loc[comparison_df[r2_col].idxmax()] if r2_col else None
        if best_row is not None:
            st.success(
                f"{best_row['Model']} predicts best for this dataset. "
                f"It has the highest R-squared score ({best_row[r2_col]:.4f}) and the lowest average error "
                f"(MAE PKR {best_row['MAE']:,.0f})."
            )

        display_df = comparison_df.copy()
        if best_row is not None:
            display_df["Result"] = np.where(display_df["Model"] == best_row["Model"], "Best predictor", "")
        display_df["MAE"]  = display_df["MAE"].apply(lambda x: f"PKR {x:,.0f}")
        display_df["RMSE"] = display_df["RMSE"].apply(lambda x: f"PKR {x:,.0f}")
        display_df["R²"]   = display_df["R²"].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption("✅ Best model (highest R²) is automatically selected for predictions above.")

st.divider()

st.subheader("Basic Project Details")
c1, c2 = st.columns(2)
with c1:
    project_type      = st.selectbox("What type of project is this?", list(project_type_map.keys()))
    project_location  = st.selectbox("Where is the project located?", list(project_location_map.keys()))
    construction_type = st.selectbox("What type of construction?", list(construction_type_map.keys()))
    foundation_type   = st.selectbox("What foundation type?", list(foundation_type_map.keys()))
with c2:
    total_area_sqft = st.number_input("Total built-up area (in square feet)", min_value=100, max_value=50000, value=2000, step=100)
    num_floors      = st.number_input("How many floors?", min_value=1, max_value=50, value=2)
    equipment_cost  = st.number_input("Estimated equipment cost (PKR) — cranes, machinery etc.", min_value=0, value=150000, step=10000)

st.divider()
st.subheader("Materials & Labor")
c3, c4 = st.columns(2)
with c3:
    material_quality = st.selectbox("What quality of materials will you use?", list(MATERIAL_QUALITY.keys()))
    material_cost_lv = st.selectbox("What are current material prices like?", list(MATERIAL_COST.keys()))
    labor_quality    = st.selectbox("What type of labor will you hire?", list(LABOR_QUALITY.keys()))
with c4:
    experience   = st.selectbox("How experienced is your contractor?", list(EXPERIENCE.keys()))
    resource_use = st.selectbox("How efficiently will resources be used?", list(RESOURCE_USE.keys()))
    inflation_lv = st.selectbox("What is the current inflation situation?", list(INFLATION.keys()))

st.divider()
st.subheader("Site & Risk Conditions")
c5, c6 = st.columns(2)
with c5:
    temperature   = st.selectbox("What is the average temperature at the site?", list(TEMPERATURE.keys()))
    rainfall      = st.selectbox("How much rainfall does the area get?", list(RAINFALL.keys()))
    weather_delay = st.selectbox("Are weather delays expected?", list(WEATHER_DELAY.keys()))
with c6:
    complexity = st.selectbox("How complex is the design?", list(COMPLEXITY.keys()))
    risk       = st.selectbox("What is the overall project risk level?", list(RISK.keys()))

st.divider()

if st.button("Estimate My Construction Cost", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "project_type"               : project_type_map[project_type],
        "project_location"           : project_location_map[project_location],
        "total_area_sqft"            : total_area_sqft,
        "num_floors"                 : num_floors,
        "construction_type"          : construction_type_map[construction_type],
        "foundation_type"            : foundation_type_map[foundation_type],
        "material_quality_index"     : MATERIAL_QUALITY[material_quality],
        "labor_cost_index"           : LABOR_QUALITY[labor_quality],
        "material_cost_index"        : MATERIAL_COST[material_cost_lv],
        "inflation_rate"             : INFLATION[inflation_lv],
        "equipment_cost"             : equipment_cost,
        "contractor_experience_years": EXPERIENCE[experience],
        "avg_temperature"            : TEMPERATURE[temperature],
        "rainfall_index"             : RAINFALL[rainfall],
        "weather_delay_days"         : WEATHER_DELAY[weather_delay],
        "complexity_score"           : COMPLEXITY[complexity],
        "risk_index"                 : RISK[risk],
        "resource_utilization_rate"  : RESOURCE_USE[resource_use],
    }])[feature_names]

    raw_prediction = float(model.predict(input_data)[0])
    minimum_area_based_cost = total_area_sqft * minimum_cost_per_sqft
    prediction = max(raw_prediction, minimum_area_based_cost, 0)

    st.success(f"### Estimated Total Project Cost: PKR {prediction:,.0f}")
    if raw_prediction < prediction:
        st.warning(
            "The model produced an unrealistically low estimate for this input, so the displayed cost "
            "was adjusted using the lowest cost per square foot seen in the training data."
        )
    r1, r2, r3 = st.columns(3)
    r1.metric("Total Estimated Cost", f"PKR {prediction:,.0f}")
    r2.metric("Cost per sqft",        f"PKR {prediction / total_area_sqft:,.0f}")
    r3.metric("Cost per floor",       f"PKR {prediction / num_floors:,.0f}")

    st.divider()

    # ── XAI tabs: SHAP + LIME ──
    st.subheader("🔍 Explainable AI (XAI) — Why this estimate?")
    shap_tab, lime_tab = st.tabs(["SHAP Explanation", "LIME Explanation"])

    # ── SHAP ──
    with shap_tab:
        st.caption("SHAP (SHapley Additive exPlanations) — Red = increases cost | Green = reduces cost")

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(input_data)

        shap_df = pd.DataFrame({
            "Feature": [FRIENDLY_NAMES.get(f, f) for f in feature_names],
            "Impact" : shap_vals[0],
        }).sort_values("Impact", key=abs, ascending=True).tail(8)

        colors  = ["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["Impact"]]
        labels  = [f"+PKR {abs(v):,.0f} ↑" if v > 0 else f"-PKR {abs(v):,.0f} ↓" for v in shap_df["Impact"]]
        max_val = shap_df["Impact"].abs().max()

        fig, ax = plt.subplots(figsize=(8, 4))
        bars    = ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors, height=0.55)
        for bar, label in zip(bars, labels):
            x = bar.get_width()
            ax.text(
                x + max_val * 0.02 if x >= 0 else x - max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                label, va="center",
                ha="left" if x >= 0 else "right",
                fontsize=8, color="#222"
            )
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.set_xlim(-max_val * 1.5, max_val * 1.5)
        ax.set_title("Top Factors Affecting Your Cost (SHAP)", fontsize=12, fontweight="bold")
        ax.xaxis.set_visible(False)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.legend(handles=[
            mpatches.Patch(color="#e74c3c", label="Increases cost"),
            mpatches.Patch(color="#2ecc71", label="Reduces cost"),
        ], fontsize=8, loc="lower right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        top_pos = shap_df[shap_df["Impact"] > 0].sort_values("Impact", ascending=False).head(2)
        top_neg = shap_df[shap_df["Impact"] < 0].sort_values("Impact").head(2)

        st.markdown("**In simple terms:**")
        for _, row in top_pos.iterrows():
            st.markdown(f"- 🔴 **{row['Feature']}** is increasing your cost")
        for _, row in top_neg.iterrows():
            st.markdown(f"- 🟢 **{row['Feature']}** is helping keep your cost down")

    # ── LIME ──
    with lime_tab:
        st.caption("LIME (Local Interpretable Model-agnostic Explanations) — explains this specific prediction locally")

        if lime_explainer is not None:
            with st.spinner("Generating LIME explanation..."):
                try:
                    lime_exp = lime_explainer.explain_instance(
                        data_row   = input_data.values[0],
                        predict_fn = model.predict,
                        num_features = 10,
                    )

                    # Build a clean bar chart from LIME results
                    lime_features = lime_exp.as_list()
                    lime_df = pd.DataFrame(lime_features, columns=["Feature", "Weight"])
                    lime_df = lime_df.sort_values("Weight", key=abs, ascending=True)

                    lime_colors = ["#e74c3c" if w > 0 else "#2ecc71" for w in lime_df["Weight"]]
                    lime_labels = [f"+{abs(w):,.0f} ↑" if w > 0 else f"-{abs(w):,.0f} ↓" for w in lime_df["Weight"]]
                    max_lime    = lime_df["Weight"].abs().max()

                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    bars2 = ax2.barh(lime_df["Feature"], lime_df["Weight"], color=lime_colors, height=0.55)
                    for bar, label in zip(bars2, lime_labels):
                        x = bar.get_width()
                        ax2.text(
                            x + max_lime * 0.02 if x >= 0 else x - max_lime * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            label, va="center",
                            ha="left" if x >= 0 else "right",
                            fontsize=8, color="#222"
                        )
                    ax2.axvline(0, color="#888", linewidth=0.8)
                    ax2.set_xlim(-max_lime * 1.5, max_lime * 1.5)
                    ax2.set_title("LIME — Local Feature Contributions", fontsize=12, fontweight="bold")
                    ax2.xaxis.set_visible(False)
                    ax2.spines[["top", "right", "bottom"]].set_visible(False)
                    ax2.legend(handles=[
                        mpatches.Patch(color="#e74c3c", label="Pushes cost higher"),
                        mpatches.Patch(color="#2ecc71", label="Pushes cost lower"),
                    ], fontsize=8, loc="lower right")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()

                    st.markdown("**LIME explains this prediction by fitting a simple linear model around your specific inputs.**")
                    st.markdown("Each bar shows how much that feature condition is pushing the estimate up or down for *your* project.")

                except Exception as e:
                    st.error(f"LIME explanation error: {e}")
        else:
            st.info("⚠️ LIME explainer not found. Please run `train_model.py` first to generate `lime_explainer.pkl`.")

    st.divider()

    # ── AI Suggestions ──
    st.subheader("💡 AI Cost-Saving Suggestions")

    top_pos_shap = shap_df[shap_df["Impact"] > 0].sort_values("Impact", ascending=False).head(2)
    top_drivers  = ", ".join(top_pos_shap["Feature"].tolist()) if not top_pos_shap.empty else "material and labor costs"

    prompt = f"""You are a construction cost advisor. A client has a {project_type} project:
- Location: {project_location}
- Area: {total_area_sqft} sqft, {num_floors} floors
- Construction: {construction_type}, Foundation: {foundation_type}
- Materials: {material_quality}, Labor: {labor_quality}
- Estimated cost: PKR {prediction:,.0f}
- Main cost drivers: {top_drivers}

Give 4 specific, practical suggestions to reduce costs while maintaining quality.
Write in simple language a non-expert builder can understand. Use bullet points. No jargon."""

    api_key = st.secrets["GROQ_API_KEY"]
    if not api_key:
        st.warning("⚠️ Add GROQ_API_KEY to .streamlit/secrets.toml to enable AI suggestions.")
    else:
        with st.spinner("Getting AI suggestions..."):
            try:
                st.info(get_ai_suggestions(prompt, api_key))
            except requests.exceptions.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Error: {e}")
