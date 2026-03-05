"""
ML Predictor Tab — Python Recommender + Salary Prediction Models
Integrates trained ML models into the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------

@st.cache_resource
def load_ml_model():
    """Load the trained Python Recommender model artifacts."""
    import joblib

    model_path = Path("python_recommender_model.joblib")
    if not model_path.exists():
        return None

    artifacts = joblib.load(model_path)
    return artifacts


@st.cache_resource
def load_salary_model():
    """Load the trained Salary Prediction model artifacts."""
    import joblib

    model_path = Path("salary_prediction_model.joblib")
    if not model_path.exists():
        return None

    artifacts = joblib.load(model_path)
    return artifacts


# ---------------------------------------------------------------------------
# HELPER: safe widget default (avoids Session State conflict warning)
# ---------------------------------------------------------------------------

def _safe_default(key, default):
    """Return {} if key is already in session_state (let Streamlit use it),
    otherwise return {'value': default} to set the initial default."""
    if key in st.session_state:
        return {}
    return {"value": default}


# ---------------------------------------------------------------------------
# FEATURE DEFINITIONS — PYTHON RECOMMENDER (must match the notebook exactly)
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age_years", "years_experience_current_role",
    "pct_time_actively_coding_at_home_or_work",
    "years_coding_for_data", "years_using_ml",
    "pct_time_gathering_data", "pct_time_cleaning_data",
    "pct_time_model_building", "pct_time_finding_insights",
    "pct_training_selftaught", "pct_training_online_courses",
    "pct_training_university", "pct_training_kaggle",
]

BINARY_LANGUAGES = {
    "Python": "uses_python",
    "R": "uses_r",
    "SQL": "uses_sql",
    "C++": "uses_cpp",
    "Java": "uses_java",
    "MATLAB": "uses_matlab",
    "Javascript": "uses_javascript",
    "Scala": "uses_scala",
    "SAS": "uses_sas",
}

BINARY_FRAMEWORKS = {
    "TensorFlow": "uses_tensorflow",
    "Keras": "uses_keras",
    "Scikit-learn": "uses_scikit-learn",
    "PyTorch": "uses_pytorch",
    "xgboost": "uses_xgboost",
    "Spark": "uses_spark",
}

BINARY_IDES = {
    "Jupyter": "ide_jupyter",
    "RStudio": "ide_rstudio",
    "PyCharm": "ide_pycharm",
    "Visual Studio": "ide_visual_studio",
    "Notepad++": "ide_notepadpp",
    "MATLAB": "ide_matlab",
}

COUNT_FEATURES = [
    "num_languages", "num_ml_frameworks", "num_ides",
    "num_viz_tools", "num_cloud_services", "num_online_platforms",
    "num_data_types",
]

HIGH_CARD_COLS = [
    "country", "current_role", "undergraduate_major", "current_industry",
    "primary_analysis_tool", "considers_self_data_scientist",
    "academics_vs_projects_view", "education_level", "gender",
]

# Dropdown options for categorical fields (most common from the dataset)
ROLE_OPTIONS = [
    "Data Scientist", "Software Engineer", "Data Analyst",
    "Research Scientist", "Student", "Business Analyst",
    "Machine Learning Engineer", "Data Engineer", "Statistician",
    "DBA/Database Engineer", "Not employed", "Other",
]

EDUCATION_OPTIONS = [
    "Doctoral degree", "Masters degree", "Bachelors degree",
    "Professional degree", "Some college/university study without earning a bachelors degree",
    "No formal education past high school", "I prefer not to answer",
]

COUNTRY_OPTIONS = [
    "United States of America", "India", "China", "Brazil", "Russia",
    "Germany", "United Kingdom of Great Britain and Northern Ireland",
    "Japan", "Canada", "France", "Australia", "Spain", "South Korea",
    "Nigeria", "Other",
]

INDUSTRY_OPTIONS = [
    "Computers/Technology", "Academics/Education", "Accounting/Finance",
    "Government", "Medical", "Military/Security", "Insurance/Risk Assessment",
    "Online Business/Internet-based Sales", "Online Service/Internet-based Services",
    "Marketing", "Manufacturing", "Retail", "Other",
]

TOOL_OPTIONS = [
    "Jupyter/IPython", "RStudio", "PyCharm", "Visual Studio / Visual Studio Code",
    "MATLAB", "Notepad++", "Sublime Text", "Other",
]

GENDER_OPTIONS = ["Male", "Female", "Prefer not to say", "Prefer to self-describe"]

DS_IDENTITY_OPTIONS = ["Definitely yes", "Probably yes", "Maybe", "Probably not", "Definitely not"]

ACADEMICS_OPTIONS = [
    "Most of what I've learned came from practical projects",
    "Most of what I've learned came from academic courses",
    "About equal \u2014 both academic and practical experiences contributed",
]


# ---------------------------------------------------------------------------
# PREDICTION FUNCTION — PYTHON RECOMMENDER
# ---------------------------------------------------------------------------

def predict_python_recommendation(features_dict: dict, artifacts: dict) -> tuple:
    """
    Build feature vector from user inputs and return (prediction_label, probability).
    """
    model = artifacts["best_model"]
    target_maps = artifacts["target_maps"]
    global_mean = artifacts["global_mean"]
    feature_columns = artifacts["feature_columns"]

    input_data = {}

    # Numeric features
    for col in NUMERIC_FEATURES:
        input_data[col] = features_dict.get(col, 0)

    # Binary features
    all_binary = list(BINARY_LANGUAGES.values()) + list(BINARY_FRAMEWORKS.values()) + list(BINARY_IDES.values())
    for col in all_binary:
        input_data[col] = features_dict.get(col, 0)

    # Count features
    for col in COUNT_FEATURES:
        input_data[col] = features_dict.get(col, 0)

    # Engineered features
    yrs_ml = features_dict.get("years_using_ml", 0)
    yrs_code = features_dict.get("years_coding_for_data", 0)
    pct_coding = features_dict.get("pct_time_actively_coding_at_home_or_work", 0)
    input_data["ml_experience_ratio"] = yrs_ml / (yrs_code + 1)
    input_data["coding_intensity"] = pct_coding * yrs_code

    # Target-encoded categoricals
    for col in HIGH_CARD_COLS:
        te_col = col + "_te"
        raw_val = features_dict.get(col, "Unknown")
        col_map = target_maps.get(col, {})
        input_data[te_col] = col_map.get(raw_val, global_mean)

    # Build DataFrame in exact column order the model expects
    input_df = pd.DataFrame([input_data])

    for c in feature_columns:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[feature_columns]

    prob = model.predict_proba(input_df)[0][1]
    label = "\u2705 YES \u2014 Recommends Python" if prob >= 0.5 else "\u274c NO \u2014 Recommends Other Language"
    return label, prob


# ---------------------------------------------------------------------------
# PRESET PROFILES — PYTHON RECOMMENDER
# ---------------------------------------------------------------------------

PRESETS = {
    "\U0001f40d Python Data Scientist": {
        "uses_python": 1, "uses_r": 0, "uses_sql": 1, "uses_cpp": 0,
        "uses_java": 0, "uses_matlab": 0, "uses_javascript": 0,
        "uses_scala": 0, "uses_sas": 0,
        "uses_tensorflow": 1, "uses_keras": 1, "uses_scikit-learn": 1,
        "uses_pytorch": 0, "uses_xgboost": 1, "uses_spark": 0,
        "ide_jupyter": 1, "ide_rstudio": 0, "ide_pycharm": 1,
        "ide_visual_studio": 0, "ide_notepadpp": 0, "ide_matlab": 0,
        "age_years": 30, "years_coding_for_data": 5, "years_using_ml": 3,
        "years_experience_current_role": 3,
        "pct_time_actively_coding_at_home_or_work": 60,
        "pct_time_gathering_data": 15, "pct_time_cleaning_data": 15,
        "pct_time_model_building": 30, "pct_time_finding_insights": 20,
        "pct_training_selftaught": 40, "pct_training_online_courses": 25,
        "pct_training_university": 15, "pct_training_kaggle": 20,
        "num_languages": 4, "num_ml_frameworks": 4, "num_ides": 3,
        "num_viz_tools": 3, "num_cloud_services": 2, "num_online_platforms": 3,
        "num_data_types": 3,
        "current_role": "Data Scientist", "education_level": "Masters degree",
        "country": "United States of America", "gender": "Male",
        "current_industry": "Computers/Technology",
        "primary_analysis_tool": "Jupyter/IPython",
        "considers_self_data_scientist": "Definitely yes",
        "academics_vs_projects_view": "Most of what I've learned came from practical projects",
        "undergraduate_major": "Computer Science",
    },
    "\U0001f4ca R Statistician": {
        "uses_python": 0, "uses_r": 1, "uses_sql": 0, "uses_cpp": 0,
        "uses_java": 0, "uses_matlab": 0, "uses_javascript": 0,
        "uses_scala": 0, "uses_sas": 1,
        "uses_tensorflow": 0, "uses_keras": 0, "uses_scikit-learn": 0,
        "uses_pytorch": 0, "uses_xgboost": 0, "uses_spark": 0,
        "ide_jupyter": 0, "ide_rstudio": 1, "ide_pycharm": 0,
        "ide_visual_studio": 0, "ide_notepadpp": 0, "ide_matlab": 0,
        "age_years": 40, "years_coding_for_data": 10, "years_using_ml": 5,
        "years_experience_current_role": 7,
        "pct_time_actively_coding_at_home_or_work": 40,
        "pct_time_gathering_data": 20, "pct_time_cleaning_data": 25,
        "pct_time_model_building": 20, "pct_time_finding_insights": 25,
        "pct_training_selftaught": 20, "pct_training_online_courses": 10,
        "pct_training_university": 60, "pct_training_kaggle": 5,
        "num_languages": 2, "num_ml_frameworks": 1, "num_ides": 1,
        "num_viz_tools": 2, "num_cloud_services": 0, "num_online_platforms": 1,
        "num_data_types": 2,
        "current_role": "Statistician", "education_level": "Doctoral degree",
        "country": "United Kingdom of Great Britain and Northern Ireland",
        "gender": "Female",
        "current_industry": "Academics/Education",
        "primary_analysis_tool": "RStudio",
        "considers_self_data_scientist": "Probably not",
        "academics_vs_projects_view": "Most of what I've learned came from academic courses",
        "undergraduate_major": "Mathematics or statistics",
    },
    "\U0001f195 Complete Beginner": {
        "uses_python": 0, "uses_r": 0, "uses_sql": 0, "uses_cpp": 0,
        "uses_java": 0, "uses_matlab": 0, "uses_javascript": 0,
        "uses_scala": 0, "uses_sas": 0,
        "uses_tensorflow": 0, "uses_keras": 0, "uses_scikit-learn": 0,
        "uses_pytorch": 0, "uses_xgboost": 0, "uses_spark": 0,
        "ide_jupyter": 0, "ide_rstudio": 0, "ide_pycharm": 0,
        "ide_visual_studio": 0, "ide_notepadpp": 0, "ide_matlab": 0,
        "age_years": 22, "years_coding_for_data": 0, "years_using_ml": 0,
        "years_experience_current_role": 0,
        "pct_time_actively_coding_at_home_or_work": 10,
        "pct_time_gathering_data": 10, "pct_time_cleaning_data": 10,
        "pct_time_model_building": 5, "pct_time_finding_insights": 5,
        "pct_training_selftaught": 80, "pct_training_online_courses": 10,
        "pct_training_university": 5, "pct_training_kaggle": 5,
        "num_languages": 0, "num_ml_frameworks": 0, "num_ides": 0,
        "num_viz_tools": 0, "num_cloud_services": 0, "num_online_platforms": 1,
        "num_data_types": 0,
        "current_role": "Student", "education_level": "Bachelors degree",
        "country": "India", "gender": "Male",
        "current_industry": "Academics/Education",
        "primary_analysis_tool": "Other",
        "considers_self_data_scientist": "Definitely not",
        "academics_vs_projects_view": "Most of what I've learned came from academic courses",
        "undergraduate_major": "Computer Science",
    },
}


# ---------------------------------------------------------------------------
# HELPER: confidence gauge chart
# ---------------------------------------------------------------------------

def _make_gauge(prob: float, title: str = "Python Recommendation Probability") -> go.Figure:
    """Create a half-donut gauge showing probability."""
    pct = prob * 100
    color = "#10b981" if prob >= 0.5 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                     "tickfont": {"color": "#a0aec0"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#2d3748",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#e94560", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
        title={"text": title,
               "font": {"color": "#cbd5e0", "size": 16}},
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="#1a1f2e",
        font={"color": "#e0e6ed"},
    )
    return fig


# ---------------------------------------------------------------------------
# MAIN TAB RENDERER
# ---------------------------------------------------------------------------

def render_ml_predictor_tab():
    """Render the full ML Predictor tab content with model selection."""

    st.markdown("### \U0001f916 Machine Learning Predictors")
    st.caption("Choose a model below to make predictions using our trained ML models on the Kaggle 2018 survey data.")

    model_tab1, model_tab2 = st.tabs([
        "\U0001f40d Python Recommender (80.1%)",
        "\U0001f4b0 Salary Predictor (66.4%)",
    ])

    with model_tab1:
        _render_python_recommender()

    with model_tab2:
        _render_salary_predictor()


# ---------------------------------------------------------------------------
# PYTHON RECOMMENDER SUB-TAB
# ---------------------------------------------------------------------------

def _render_python_recommender():
    """Render the Python Recommender model UI."""

    artifacts = load_ml_model()

    if artifacts is None:
        st.error(
            "\u26a0\ufe0f Model file `python_recommender_model.joblib` not found. "
            "Please ensure the trained model is in the same directory as `app.py`."
        )
        return

    results = artifacts.get("results", {})

    # ── Model Performance KPIs ────────────────────────────────────────────
    st.markdown("### \U0001f3c6 Model Performance Summary")
    st.markdown(
        "The **Python Recommender** is our best ML model \u2014 trained on the Kaggle 2018 survey "
        "to predict whether a data professional would recommend **Python** as the first language to learn."
    )

    if results:
        best_name = max(results, key=lambda k: results[k].get("Accuracy", 0))
        best = results[best_name]
    else:
        best_name = "Tuned XGBoost"
        best = {"Accuracy": 0.8011, "F1": 0.8798, "ROC-AUC": 0.7715,
                "Recall": 0.9577, "Precision": 0.8137}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{best.get('Accuracy', 0):.1%}")
    c2.metric("F1 Score", f"{best.get('F1', 0):.4f}")
    c3.metric("ROC-AUC", f"{best.get('ROC-AUC', 0):.4f}")
    c4.metric("Recall", f"{best.get('Recall', 0):.4f}")
    c5.metric("Precision", f"{best.get('Precision', 0):.4f}")

    st.caption(f"Best model: **{best_name}** \u2014 Voting Ensemble (LR + RF + GB + XGBoost with target encoding)")

    if results:
        with st.expander("\U0001f4ca All Model Results Comparison", expanded=False):
            rows = []
            for name, m in sorted(results.items(), key=lambda x: x[1].get("Accuracy", 0), reverse=True):
                rows.append({
                    "Model": name,
                    "Accuracy": f"{m.get('Accuracy', 0):.2%}",
                    "Precision": f"{m.get('Precision', 0):.4f}",
                    "Recall": f"{m.get('Recall', 0):.4f}",
                    "F1": f"{m.get('F1', 0):.4f}",
                    "ROC-AUC": f"{m.get('ROC-AUC', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            model_names = [r["Model"] for r in rows]
            accuracies = [results[n].get("Accuracy", 0) for n in model_names]
            f1_scores = [results[n].get("F1", 0) for n in model_names]

            fig_comp = go.Figure(data=[
                go.Bar(name="Accuracy", x=model_names, y=accuracies, marker_color="#e94560"),
                go.Bar(name="F1 Score", x=model_names, y=f1_scores, marker_color="#3498db"),
            ])
            fig_comp.update_layout(
                barmode="group", title="Model Comparison",
                yaxis_title="Score", xaxis_tickangle=-25,
                height=400, plot_bgcolor="#111827", paper_bgcolor="#1a1f2e",
                legend=dict(font=dict(color="#e0e6ed")),
                font=dict(color="#e0e6ed"),
            )
            st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── Usage Guide ────────────────────────────────────────────────────────
    with st.expander("\U0001f4d8 **How to Use This ML Predictor \u2014 Read This First!**", expanded=True):
        st.markdown("""
        This tab lets you **predict whether a data professional would recommend Python** as the
        first programming language to learn. The model was trained on **23,858 real survey responses**
        from the Kaggle 2018 ML & Data Science Survey.

        #### \U0001f680 Quick Start (3 Steps)
        1. **Pick a preset** (or fill the form manually)
        2. **Adjust any fields** to match the person you want to predict for
        3. **Click "Predict Now"** \u2014 See the result with confidence % and a gauge chart

        #### \U0001f4cb What Each Section Means

        | Section | What to Select | How It Affects Prediction |
        |---------|---------------|---------------------------|
        | **Programming Languages** | Check the languages this person currently uses | Python/R users have strong signals |
        | **ML Frameworks** | Check the ML/DL libraries this person uses | TensorFlow/Keras/Scikit-learn users lean towards Python |
        | **IDEs Used** | Check the code editors/IDEs this person uses | Jupyter \u2192 Python signal; RStudio \u2192 R signal |
        | **Tool Counts** | Slide to set how many viz tools, cloud services etc. | More tools = more experienced |
        | **Experience & Background** | Set age, years of experience in ML/coding | More ML experience \u2192 stronger Python signal |
        | **Time Allocation** | % of work time on each activity | More model-building time \u2192 Python |
        | **Training Sources** | % of learning from each source | Self-taught & Kaggle learners lean Python |
        | **Profile Details** | Select role, education, country, industry etc. | Data Scientists \u2192 Python; Statisticians \u2192 R |

        #### \U0001f4a1 Tips
        - **Languages, Frameworks & IDEs counts auto-sync** \u2014 The model automatically counts how many you checked
        - **Training % and Time % don't need to add up to 100%**
        - **Try the presets first** to see how the prediction changes
        - **The model is ~80% accurate** \u2014 It reflects real-world patterns from 2018 survey data
        """)

    # ── Interactive Prediction ─────────────────────────────────────────────
    st.markdown("### \U0001f52e Try It: Predict for a New Person")
    st.markdown("Fill in the profile below and click **Predict** to see if this person would recommend Python.")

    st.markdown("##### \u26a1 Quick Presets")
    st.caption("Click a preset to instantly fill ALL fields below with a realistic profile. Then click **Predict Now**.")
    preset_cols = st.columns(len(PRESETS))

    _WIDGET_KEY_MAP = {
        "uses_python": "lang_uses_python", "uses_r": "lang_uses_r", "uses_sql": "lang_uses_sql",
        "uses_cpp": "lang_uses_cpp", "uses_java": "lang_uses_java", "uses_matlab": "lang_uses_matlab",
        "uses_javascript": "lang_uses_javascript", "uses_scala": "lang_uses_scala", "uses_sas": "lang_uses_sas",
        "uses_tensorflow": "fw_uses_tensorflow", "uses_keras": "fw_uses_keras",
        "uses_scikit-learn": "fw_uses_scikit-learn", "uses_pytorch": "fw_uses_pytorch",
        "uses_xgboost": "fw_uses_xgboost", "uses_spark": "fw_uses_spark",
        "ide_jupyter": "ide_ide_jupyter", "ide_rstudio": "ide_ide_rstudio",
        "ide_pycharm": "ide_ide_pycharm", "ide_visual_studio": "ide_ide_visual_studio",
        "ide_notepadpp": "ide_ide_notepadpp", "ide_matlab": "ide_ide_matlab",
        "age_years": "sl_age", "years_coding_for_data": "sl_yrs_code",
        "years_using_ml": "sl_yrs_ml", "years_experience_current_role": "sl_yrs_role",
        "pct_time_actively_coding_at_home_or_work": "sl_pct_code",
        "pct_time_gathering_data": "sl_pct_gather", "pct_time_cleaning_data": "sl_pct_clean",
        "pct_time_model_building": "sl_pct_model", "pct_time_finding_insights": "sl_pct_insights",
        "pct_training_selftaught": "sl_tr_self", "pct_training_online_courses": "sl_tr_online",
        "pct_training_university": "sl_tr_uni", "pct_training_kaggle": "sl_tr_kaggle",
        "num_viz_tools": "sl_num_viz", "num_cloud_services": "sl_num_cloud",
        "num_online_platforms": "sl_num_op", "num_data_types": "sl_num_dt",
        "current_role": "sel_role", "education_level": "sel_edu",
        "country": "sel_country", "gender": "sel_gender",
        "current_industry": "sel_industry", "primary_analysis_tool": "sel_tool",
        "considers_self_data_scientist": "sel_ds", "academics_vs_projects_view": "sel_acad",
    }

    for i, (preset_name, preset_data) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            if st.button(preset_name, use_container_width=True, key=f"preset_{i}"):
                for data_key, widget_key in _WIDGET_KEY_MAP.items():
                    if data_key in preset_data:
                        val = preset_data[data_key]
                        if widget_key.startswith(("lang_", "fw_", "ide_")):
                            st.session_state[widget_key] = bool(val)
                        else:
                            st.session_state[widget_key] = val
                st.rerun()

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### \U0001f527 Programming Languages")
        st.caption("Check all programming languages this person currently uses.")
        lang_cols = st.columns(3)
        lang_vals = {}
        for i, (display, col_name) in enumerate(BINARY_LANGUAGES.items()):
            with lang_cols[i % 3]:
                lang_vals[col_name] = int(st.checkbox(display, key=f"lang_{col_name}", **_safe_default(f"lang_{col_name}", False)))

        st.markdown("##### \U0001f9e0 ML Frameworks")
        st.caption("Check all ML/deep learning libraries this person has used.")
        fw_cols = st.columns(3)
        fw_vals = {}
        for i, (display, col_name) in enumerate(BINARY_FRAMEWORKS.items()):
            with fw_cols[i % 3]:
                fw_vals[col_name] = int(st.checkbox(display, key=f"fw_{col_name}", **_safe_default(f"fw_{col_name}", False)))

        st.markdown("##### \U0001f4bb IDEs Used")
        st.caption("Check all code editors / development environments this person uses.")
        ide_cols = st.columns(3)
        ide_vals = {}
        for i, (display, col_name) in enumerate(BINARY_IDES.items()):
            with ide_cols[i % 3]:
                ide_vals[col_name] = int(st.checkbox(display, key=f"ide_{col_name}", **_safe_default(f"ide_{col_name}", False)))

        num_languages = sum(lang_vals.values())
        num_ml_frameworks = sum(fw_vals.values())
        num_ides = sum(ide_vals.values())

        st.markdown(f"*Auto-counted:* **{num_languages}** languages \u00b7 **{num_ml_frameworks}** ML frameworks \u00b7 **{num_ides}** IDEs")

        st.markdown("##### \U0001f4e6 Additional Tool Counts")
        st.caption("Set how many visualization tools, cloud platforms, etc. this person uses overall.")
        tc1, tc2 = st.columns(2)
        with tc1:
            num_viz_tools = st.slider("# Visualization Tools", 0, 10, key="sl_num_viz", **_safe_default("sl_num_viz", 1))
            num_cloud_services = st.slider("# Cloud Services", 0, 10, key="sl_num_cloud", **_safe_default("sl_num_cloud", 0))
        with tc2:
            num_online_platforms = st.slider("# Online Learning Platforms", 0, 10, key="sl_num_op", **_safe_default("sl_num_op", 1))
            num_data_types = st.slider("# Data Types worked with", 0, 10, key="sl_num_dt", **_safe_default("sl_num_dt", 2))

    with col_right:
        st.markdown("##### \U0001f4da Experience & Background")
        st.caption("Set the person's age and years of professional experience.")
        age = st.slider("Age (years)", 18, 70, key="sl_age", **_safe_default("sl_age", 28))
        years_coding = st.slider("Years coding for data", 0, 30, key="sl_yrs_code", **_safe_default("sl_yrs_code", 2))
        years_ml = st.slider("Years using ML", 0, 20, key="sl_yrs_ml", **_safe_default("sl_yrs_ml", 1))
        years_role = st.slider("Years in current role", 0, 30, key="sl_yrs_role", **_safe_default("sl_yrs_role", 2))

        st.markdown("##### \u23f1\ufe0f Time Allocation (%)")
        st.caption("Estimate what % of work time is spent on each activity (don't need to sum to 100).")
        pct_coding = st.slider("% Time actively coding", 0, 100, key="sl_pct_code", **_safe_default("sl_pct_code", 40))
        pct_gather = st.slider("% Time gathering data", 0, 100, key="sl_pct_gather", **_safe_default("sl_pct_gather", 15))
        pct_clean = st.slider("% Time cleaning data", 0, 100, key="sl_pct_clean", **_safe_default("sl_pct_clean", 15))
        pct_model = st.slider("% Time model building", 0, 100, key="sl_pct_model", **_safe_default("sl_pct_model", 20))
        pct_insights = st.slider("% Time finding insights", 0, 100, key="sl_pct_insights", **_safe_default("sl_pct_insights", 15))

        st.markdown("##### \U0001f4d6 Training Sources (%)")
        st.caption("Estimate what % of knowledge came from each source (don't need to sum to 100).")
        tr1, tr2 = st.columns(2)
        with tr1:
            pct_self = st.slider("Self-taught", 0, 100, key="sl_tr_self", **_safe_default("sl_tr_self", 40))
            pct_online = st.slider("Online courses", 0, 100, key="sl_tr_online", **_safe_default("sl_tr_online", 20))
        with tr2:
            pct_uni = st.slider("University", 0, 100, key="sl_tr_uni", **_safe_default("sl_tr_uni", 20))
            pct_kaggle = st.slider("Kaggle", 0, 100, key="sl_tr_kaggle", **_safe_default("sl_tr_kaggle", 10))

        st.markdown("##### \U0001f3f7\ufe0f Profile Details")
        st.caption("Select role, education, country, and other demographics.")
        role = st.selectbox("Current Role", ROLE_OPTIONS, key="sel_role", index=None if "sel_role" in st.session_state else 0)
        education = st.selectbox("Education Level", EDUCATION_OPTIONS, key="sel_edu", index=None if "sel_edu" in st.session_state else 2)
        country = st.selectbox("Country", COUNTRY_OPTIONS, key="sel_country", index=None if "sel_country" in st.session_state else 0)
        gender = st.selectbox("Gender", GENDER_OPTIONS, key="sel_gender", index=None if "sel_gender" in st.session_state else 0)
        industry = st.selectbox("Industry", INDUSTRY_OPTIONS, key="sel_industry", index=None if "sel_industry" in st.session_state else 0)
        tool = st.selectbox("Primary Analysis Tool", TOOL_OPTIONS, key="sel_tool", index=None if "sel_tool" in st.session_state else 0)
        ds_identity = st.selectbox("Consider yourself a Data Scientist?", DS_IDENTITY_OPTIONS, key="sel_ds", index=None if "sel_ds" in st.session_state else 0)
        acad_view = st.selectbox("Academics vs Projects", ACADEMICS_OPTIONS, key="sel_acad", index=None if "sel_acad" in st.session_state else 0)

    # ── Predict Button ────────────────────────────────────────────────────
    st.markdown("---")

    if st.button("\U0001f52e **Predict Now**", use_container_width=True, type="primary"):
        features = {}
        features.update(lang_vals)
        features.update(fw_vals)
        features.update(ide_vals)

        features["age_years"] = age
        features["years_coding_for_data"] = years_coding
        features["years_using_ml"] = years_ml
        features["years_experience_current_role"] = years_role
        features["pct_time_actively_coding_at_home_or_work"] = pct_coding
        features["pct_time_gathering_data"] = pct_gather
        features["pct_time_cleaning_data"] = pct_clean
        features["pct_time_model_building"] = pct_model
        features["pct_time_finding_insights"] = pct_insights
        features["pct_training_selftaught"] = pct_self
        features["pct_training_online_courses"] = pct_online
        features["pct_training_university"] = pct_uni
        features["pct_training_kaggle"] = pct_kaggle

        features["num_languages"] = num_languages
        features["num_ml_frameworks"] = num_ml_frameworks
        features["num_ides"] = num_ides
        features["num_viz_tools"] = num_viz_tools
        features["num_cloud_services"] = num_cloud_services
        features["num_online_platforms"] = num_online_platforms
        features["num_data_types"] = num_data_types

        features["current_role"] = role
        features["education_level"] = education
        features["country"] = country
        features["gender"] = gender
        features["current_industry"] = industry
        features["primary_analysis_tool"] = tool
        features["considers_self_data_scientist"] = ds_identity
        features["academics_vs_projects_view"] = acad_view
        features["undergraduate_major"] = "Computer Science"

        label, prob = predict_python_recommendation(features, artifacts)

        st.session_state["ml_pred_label"] = label
        st.session_state["ml_pred_prob"] = prob
        st.session_state["ml_pred_factors"] = {
            "uses_python": lang_vals.get("uses_python", 0),
            "uses_r": lang_vals.get("uses_r", 0),
            "uses_dl": fw_vals.get("uses_tensorflow", 0) or fw_vals.get("uses_keras", 0) or fw_vals.get("uses_pytorch", 0),
            "uses_sklearn": fw_vals.get("uses_scikit-learn", 0),
            "uses_jupyter": ide_vals.get("ide_jupyter", 0),
            "uses_rstudio": ide_vals.get("ide_rstudio", 0),
            "years_ml": years_ml,
            "num_languages": num_languages,
            "pct_self": pct_self,
        }

    # ── Display Result ────────────────────────────────────────────────────
    if "ml_pred_label" in st.session_state:
        label = st.session_state["ml_pred_label"]
        prob = st.session_state["ml_pred_prob"]
        finfo = st.session_state["ml_pred_factors"]

        st.markdown("### \U0001f3af Prediction Result")

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if prob >= 0.5:
                st.success(f"### {label}")
                st.markdown(f"**Confidence:** {prob:.1%} likely to recommend Python")
            else:
                st.error(f"### {label}")
                st.markdown(f"**Confidence:** {1 - prob:.1%} likely to recommend another language")

            st.markdown("##### \U0001f4a1 Key Factors in This Prediction")
            factors = []
            if finfo.get("uses_python"):
                factors.append("\U0001f40d Already uses Python")
            if finfo.get("uses_r") and not finfo.get("uses_python"):
                factors.append("\U0001f4ca Uses R but not Python")
            if finfo.get("uses_dl"):
                factors.append("\U0001f9e0 Uses deep learning frameworks")
            if finfo.get("uses_sklearn"):
                factors.append("\U0001f4c8 Uses Scikit-learn")
            if finfo.get("uses_jupyter"):
                factors.append("\U0001f4d3 Uses Jupyter notebooks")
            if finfo.get("uses_rstudio") and not finfo.get("uses_jupyter"):
                factors.append("\U0001f4ca Uses RStudio (R ecosystem)")
            if finfo.get("years_ml", 0) >= 3:
                factors.append(f"\u23f0 {finfo['years_ml']} years ML experience")
            if finfo.get("num_languages", 0) >= 4:
                factors.append(f"\U0001f527 Uses {finfo['num_languages']} languages (polyglot)")
            if finfo.get("pct_self", 0) >= 50:
                factors.append("\U0001f4da Primarily self-taught")
            if not factors:
                factors.append("\U0001f4cb Profile features analyzed holistically by the model")

            for f in factors[:6]:
                st.markdown(f"- {f}")

        with res_col2:
            gauge = _make_gauge(prob)
            st.plotly_chart(gauge, use_container_width=True)

    # ── Model Details ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### \U0001f4d6 Model Details")

    with st.expander("\u2139\ufe0f About This Model", expanded=False):
        st.markdown("""
        **Problem:** Predict whether a data professional would recommend **Python** as the first programming language to learn.

        **Dataset:** Kaggle 2018 Machine Learning & Data Science Survey (23,858 respondents)

        **Target:** Binary \u2014 *Recommends Python* (76%) vs *Recommends Other Language* (24%)

        **Features (97 total):** 13 numeric + 9 binary tool flags + 6 binary framework flags + 6 binary IDE flags + 7 count features + 2 engineered + 9 target-encoded categoricals

        **Best Model:** Voting Ensemble (Logistic Regression + Random Forest + Gradient Boosting + XGBoost) with target encoding and StandardScaler.

        **Training:** 80/20 stratified split, 5-fold cross-validation, RandomizedSearchCV for XGBoost tuning.
        """)

    with st.expander("\U0001f511 Feature Importance (Top 20)", expanded=False):
        model = artifacts.get("best_model")
        feature_cols = artifacts.get("feature_columns", [])

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True).tail(20)

            fig_fi = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                title="Top 20 Features \u2014 Tuned XGBoost",
                color="Importance", color_continuous_scale="Teal",
            )
            fig_fi.update_layout(
                height=550, margin=dict(l=220, r=40, t=60, b=40),
                coloraxis_showscale=False,
                plot_bgcolor="#111827", paper_bgcolor="#1a1f2e",
                font=dict(color="#e0e6ed"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    with st.expander("\U0001f4cb How It Works", expanded=False):
        st.markdown("""
        **Step-by-step pipeline:**

        1. **Data Cleaning** \u2014 Handle missing values, remove respondents with no language recommendation
        2. **Feature Engineering** \u2014 Create binary flags from multi-select columns, count features, experience ratios
        3. **Target Encoding** \u2014 Replace high-cardinality categoricals (country, role) with smoothed target means
        4. **Scaling** \u2014 StandardScaler for distance-based models
        5. **Model Training** \u2014 Train 5 base models + tuned XGBoost + Voting & Stacking ensembles
        6. **Evaluation** \u2014 Accuracy, F1, ROC-AUC, Precision, Recall on held-out 20% test set
        7. **Prediction** \u2014 New profiles are transformed through the same pipeline and scored

        **Why Python wins so often:** ~76% of the dataset recommends Python, so the model has a
        strong prior. But it genuinely learns patterns \u2014 R users, RStudio users, and those in
        statistics roles are more likely to recommend R instead.
        """)


# ===========================================================================
# SALARY PREDICTOR SUB-TAB  (5-class Stacking Ensemble \u2014 66.4% accuracy)
# ===========================================================================

SALARY_COUNTRY_OPTIONS = [
    "United States of America", "India", "China", "Brazil", "Russia",
    "Germany", "United Kingdom of Great Britain and Northern Ireland",
    "Japan", "Canada", "France", "Australia", "Spain", "South Korea",
    "Nigeria", "Other",
]

SALARY_ROLE_OPTIONS = [
    "Data Scientist", "Software Engineer", "Data Analyst",
    "Research Scientist", "Student", "Business Analyst",
    "Data Engineer", "Statistician", "DBA/Database Engineer",
    "Manager", "Consultant", "Chief Officer",
    "Product/Project Manager", "Research Assistant",
    "Marketing Analyst", "Developer Advocate", "Other",
]

SALARY_EDUCATION_OPTIONS = [
    "Bachelor's degree", "Master's degree", "Doctoral degree",
    "Professional degree",
    "Some college/university study without earning a bachelor's degree",
    "No formal education past high school", "I prefer not to answer",
]

SALARY_LANGUAGE_OPTIONS = [
    "Python", "R", "SQL", "Java", "C/C++", "MATLAB", "Javascript/Typescript",
    "Scala", "Bash", "SAS/STATA", "Go", "Julia", "C#/.NET",
    "Ruby", "PHP", "Visual Basic/VBA", "Other",
]

SALARY_INDUSTRY_OPTIONS = [
    "Computers/Technology", "Academics/Education", "Accounting/Finance",
    "Government/Public Service", "Medical/Pharmaceutical", "Military/Security/Defense",
    "Insurance/Risk Assessment", "Online Business/Internet-based Sales",
    "Online Service/Internet-based Services", "Marketing/CRM",
    "Manufacturing/Fabrication", "Retail/Sales", "Energy/Mining",
    "Broadcasting/Communications", "Shipping/Transportation",
    "Hospitality/Entertainment/Sports", "Non-profit/Service",
    "I am a student", "Other",
]

SALARY_DS_IDENTITY_OPTIONS = [
    "Definitely yes", "Probably yes", "Maybe", "Probably not", "Definitely not", "Unknown",
]

SALARY_GROUPS_ORDERED = [
    "Low ($0-30k)", "Mid-Low ($30-60k)", "Mid ($60-100k)",
    "Mid-High ($100-150k)", "High ($150k+)",
]

SALARY_GROUP_MIDPOINTS = {
    "Low ($0-30k)": 15000, "Mid-Low ($30-60k)": 45000,
    "Mid ($60-100k)": 80000, "Mid-High ($100-150k)": 125000,
    "High ($150k+)": 225000,
}

SALARY_GROUP_COLORS = {
    "Low ($0-30k)": "#ef4444", "Mid-Low ($30-60k)": "#f59e0b",
    "Mid ($60-100k)": "#3b82f6", "Mid-High ($100-150k)": "#8b5cf6",
    "High ($150k+)": "#10b981",
}

SALARY_PRESETS = {
    "\U0001f468\u200d\U0001f4bb Senior DS (USA)": {
        "sal_age": 35, "sal_yrs_role": 8,
        "sal_country": "United States of America",
        "sal_education": "Master's degree",
        "sal_role": "Data Scientist",
        "sal_num_lang": 5, "sal_num_fw": 4,
        "sal_yrs_code": 8, "sal_yrs_ml": 5,
        "sal_lang": "Python", "sal_industry": "Computers/Technology",
        "sal_ds_id": "Definitely yes",
    },
    "\U0001f393 Fresh Graduate (India)": {
        "sal_age": 23, "sal_yrs_role": 0,
        "sal_country": "India",
        "sal_education": "Bachelor's degree",
        "sal_role": "Student",
        "sal_num_lang": 2, "sal_num_fw": 1,
        "sal_yrs_code": 1, "sal_yrs_ml": 0,
        "sal_lang": "Python", "sal_industry": "Academics/Education",
        "sal_ds_id": "Probably not",
    },
    "\U0001f1ec\U0001f1e7 Manager (UK)": {
        "sal_age": 42, "sal_yrs_role": 12,
        "sal_country": "United Kingdom of Great Britain and Northern Ireland",
        "sal_education": "Master's degree",
        "sal_role": "Manager",
        "sal_num_lang": 3, "sal_num_fw": 2,
        "sal_yrs_code": 10, "sal_yrs_ml": 4,
        "sal_lang": "SQL", "sal_industry": "Accounting/Finance",
        "sal_ds_id": "Maybe",
    },
}


def _make_salary_bar(probabilities: dict) -> go.Figure:
    """Create a horizontal bar chart showing probability for each salary group."""
    bands = SALARY_GROUPS_ORDERED
    probs = [probabilities.get(b, 0) for b in bands]
    max_prob = max(probs) if probs else 0

    colors = [SALARY_GROUP_COLORS.get(b, "#3498db") if p == max_prob else "#64748b" for b, p in zip(bands, probs)]

    fig = go.Figure(go.Bar(
        x=probs, y=[f"\U0001f4b5 {b}" for b in bands], orientation="h",
        marker_color=colors,
        text=[f"{p:.1%}" for p in probs],
        textposition="auto",
    ))
    fig.update_layout(
        title="Predicted Salary Group Probabilities",
        xaxis_title="Probability",
        height=350,
        margin=dict(l=180, r=40, t=60, b=40),
        plot_bgcolor="#111827", paper_bgcolor="#1a1f2e",
        font=dict(color="#e0e6ed"),
    )
    return fig


def _render_salary_predictor():
    """Render the Salary Prediction model UI (5-class Stacking Ensemble)."""

    artifacts = load_salary_model()

    if artifacts is None:
        st.error(
            "\u26a0\ufe0f Model file `salary_prediction_model.joblib` not found. "
            "Please ensure the trained model is in the same directory as `app.py`."
        )
        return

    cls_model = artifacts["classification_model"]
    reg_model = artifacts["regression_model"]
    scaler = artifacts["scaler"]
    label_encoders = artifacts["label_encoders"]
    salary_le = artifacts["salary_label_encoder"]
    feature_columns = artifacts["feature_columns"]

    # ── Model Info ────────────────────────────────────────────────────────
    st.markdown("### \U0001f4b0 Salary Band Predictor")
    st.markdown(
        "Predict the **yearly compensation range (USD)** for a data professional based on their "
        "role, experience, education, country, and toolset. Trained on **Kaggle 2018 survey** data."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", "Stacking Ensemble")
    c2.metric("Accuracy", "66.4%")
    c3.metric("Target", "5 Salary Groups")
    c4.metric("Features", "12")

    st.markdown("---")

    with st.expander("\U0001f4d8 **How to Use the Salary Predictor**", expanded=False):
        st.markdown("""
        #### \U0001f680 Quick Start
        1. **Click a preset** or fill the form manually
        2. **Click "\U0001f4b0 Predict Salary"**
        3. See the predicted salary group + probability chart

        #### \U0001f4ca Salary Groups
        | Group | Range | Description |
        |-------|-------|-------------|
        | \U0001f534 Low | $0\u201330k | Entry-level, students, developing countries |
        | \U0001f7e1 Mid-Low | $30\u201360k | Junior roles, mid-cost countries |
        | \U0001f535 Mid | $60\u2013100k | Mid-career professionals |
        | \U0001f7e3 Mid-High | $100\u2013150k | Senior roles, high-cost countries |
        | \U0001f7e2 High | $150k+ | Principal/executive, USA tech hubs |

        #### \U0001f4cb Feature Descriptions
        | Input | Effect on Salary |
        |-------|------------------|
        | **Age / Years in Role** | More experience \u2192 higher salary |
        | **Years Coding / ML** | Technical depth \u2192 higher band |
        | **Country** | USA/UK/Australia \u2192 higher; India/Brazil \u2192 lower |
        | **Education** | Doctoral/Master's \u2192 higher salary |
        | **Role** | Chief Officer/Manager \u2192 highest; Student \u2192 lowest |
        | **Primary Language** | Python/R users have different salary patterns |
        | **Industry** | Tech/Finance \u2192 higher; Education \u2192 lower |
        | **DS Identity** | Stronger identification \u2192 higher salary |
        | **# Languages / Frameworks** | More tools \u2192 slightly higher |
        """)

    # ── Presets ───────────────────────────────────────────────────────────
    st.markdown("##### \u26a1 Quick Presets")
    sal_preset_cols = st.columns(len(SALARY_PRESETS))
    for i, (preset_name, preset_data) in enumerate(SALARY_PRESETS.items()):
        with sal_preset_cols[i]:
            if st.button(preset_name, use_container_width=True, key=f"sal_preset_{i}"):
                for widget_key, val in preset_data.items():
                    st.session_state[widget_key] = val
                st.rerun()

    st.markdown("---")

    # ── Input Form ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### \U0001f4da Experience")
        sal_age = st.slider("Age (years)", 18, 70, key="sal_age", **_safe_default("sal_age", 28))
        sal_yrs = st.slider("Years in current role", 0, 30, key="sal_yrs_role", **_safe_default("sal_yrs_role", 3))
        sal_yrs_code = st.slider("Years coding for data", 0, 30, key="sal_yrs_code", **_safe_default("sal_yrs_code", 2))
        sal_yrs_ml = st.slider("Years using ML", 0, 20, key="sal_yrs_ml", **_safe_default("sal_yrs_ml", 1))
        sal_num_lang = st.slider("# Programming languages used", 0, 15, key="sal_num_lang", **_safe_default("sal_num_lang", 3))
        sal_num_fw = st.slider("# ML frameworks used", 0, 15, key="sal_num_fw", **_safe_default("sal_num_fw", 2))

    with col2:
        st.markdown("##### \U0001f3f7\ufe0f Profile")
        sal_country = st.selectbox("Country", SALARY_COUNTRY_OPTIONS, key="sal_country",
                                   index=None if "sal_country" in st.session_state else 0)
        sal_education = st.selectbox("Education Level", SALARY_EDUCATION_OPTIONS, key="sal_education",
                                     index=None if "sal_education" in st.session_state else 0)
        sal_role = st.selectbox("Current Role", SALARY_ROLE_OPTIONS, key="sal_role",
                                index=None if "sal_role" in st.session_state else 0)
        sal_lang = st.selectbox("Primary Programming Language", SALARY_LANGUAGE_OPTIONS, key="sal_lang",
                                index=None if "sal_lang" in st.session_state else 0)
        sal_industry = st.selectbox("Industry", SALARY_INDUSTRY_OPTIONS, key="sal_industry",
                                    index=None if "sal_industry" in st.session_state else 0)
        sal_ds_id = st.selectbox("Consider yourself a Data Scientist?", SALARY_DS_IDENTITY_OPTIONS, key="sal_ds_id",
                                 index=None if "sal_ds_id" in st.session_state else 0)

    # ── Predict ───────────────────────────────────────────────────────────
    st.markdown("---")

    if st.button("\U0001f4b0 **Predict Salary**", use_container_width=True, type="primary"):
        def _encode_safe(le_dict, name, value):
            try:
                return le_dict[name].transform([value])[0]
            except (ValueError, KeyError):
                return 0

        country_enc = _encode_safe(label_encoders, "country", sal_country)
        edu_enc = _encode_safe(label_encoders, "education_level", sal_education)
        role_enc = _encode_safe(label_encoders, "current_role", sal_role)
        lang_enc = _encode_safe(label_encoders, "primary_programming_language", sal_lang)
        industry_enc = _encode_safe(label_encoders, "current_industry", sal_industry)
        ds_enc = _encode_safe(label_encoders, "considers_self_data_scientist", sal_ds_id)

        # feature_columns order: age_years, years_experience_current_role, country_encoded,
        # education_level_encoded, current_role_encoded, num_programming_languages,
        # num_ml_frameworks, years_coding_for_data, years_using_ml,
        # primary_programming_language_encoded, current_industry_encoded,
        # considers_self_data_scientist_encoded
        features = np.array([[
            sal_age, sal_yrs, country_enc, edu_enc, role_enc,
            sal_num_lang, sal_num_fw, sal_yrs_code, sal_yrs_ml,
            lang_enc, industry_enc, ds_enc,
        ]])

        features_scaled = scaler.transform(features)

        pred_encoded = cls_model.predict(features_scaled)[0]
        pred_band = salary_le.inverse_transform([pred_encoded])[0]

        pred_proba = cls_model.predict_proba(features_scaled)[0]
        band_probs = {}
        for idx, prob in enumerate(pred_proba):
            band_name = salary_le.inverse_transform([idx])[0]
            band_probs[band_name] = prob

        reg_pred = reg_model.predict(features_scaled)[0]

        st.session_state["sal_pred_band"] = pred_band
        st.session_state["sal_pred_probs"] = band_probs
        st.session_state["sal_pred_reg"] = reg_pred
        st.session_state["sal_pred_profile"] = {
            "country": sal_country, "role": sal_role, "education": sal_education,
            "age": sal_age, "years": sal_yrs, "language": sal_lang,
            "industry": sal_industry,
        }

    # ── Display Result ────────────────────────────────────────────────────
    if "sal_pred_band" in st.session_state:
        pred_band = st.session_state["sal_pred_band"]
        band_probs = st.session_state["sal_pred_probs"]
        reg_pred = st.session_state["sal_pred_reg"]
        profile = st.session_state["sal_pred_profile"]

        st.markdown("### \U0001f3af Salary Prediction Result")

        confidence = band_probs.get(pred_band, 0)

        res1, res2, res3 = st.columns(3)
        with res1:
            st.success(f"### \U0001f4b5 {pred_band}")
            st.caption("Predicted Salary Group (USD/year)")
        with res2:
            st.info(f"### \U0001f4b2 ${reg_pred:,.0f}")
            st.caption("Regression Estimate (USD/year)")
        with res3:
            st.warning(f"### \U0001f3af {confidence:.1%}")
            st.caption("Confidence in Top Group")

        st.markdown(f"**Profile:** {profile['role']} \u00b7 {profile['education']} \u00b7 {profile['country']} \u00b7 "
                    f"Age {profile['age']} \u00b7 {profile['years']} yrs experience \u00b7 {profile.get('language', 'N/A')} \u00b7 {profile.get('industry', 'N/A')}")

        fig_sal = _make_salary_bar(band_probs)
        st.plotly_chart(fig_sal, use_container_width=True)

        sorted_bands = sorted(band_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        st.markdown("##### \U0001f3c5 Top 3 Most Likely Salary Groups")
        for rank, (band, prob) in enumerate(sorted_bands, 1):
            emoji = ["\U0001f947", "\U0001f948", "\U0001f949"][rank - 1]
            st.markdown(f"{emoji} **{band}** \u2014 {prob:.1%} probability")

    # ── Model Details ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("\u2139\ufe0f About the Salary Model", expanded=False):
        st.markdown("""
        **Problem:** Predict the yearly compensation group (5 ranges from $0-30k to $150k+)

        **Dataset:** Kaggle 2018 ML & Data Science Survey

        **Features (12):**
        - Age, Years in current role, Years coding for data, Years using ML
        - Country (label-encoded, 58 countries)
        - Education level (label-encoded, 7 levels)
        - Current role (label-encoded, 20 roles)
        - Primary programming language (label-encoded, 18 languages)
        - Current industry (label-encoded, 19 industries)
        - Considers self data scientist (label-encoded, 6 levels)
        - # Programming languages used, # ML frameworks used

        **Models:** Stacking Ensemble (RF + GB + XGBoost \u2192 Logistic Regression meta-learner) for classification + Gradient Boosting Regressor for salary estimation

        **Accuracy:** 66.4% on 5-class classification (vs ~36.8% on the original 18-band approach)

        **Note:** Salary prediction is inherently difficult \u2014 many factors (company size, negotiation,
        stock options) aren't captured in the survey. Grouping into 5 bands significantly improves accuracy.
        """)
