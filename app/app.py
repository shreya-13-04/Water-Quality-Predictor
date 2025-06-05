# ----------------------------------------
# Imports
# ----------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import joblib
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from streamlit_folium import folium_static



# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="wide")

# ----------------------------------------
# Load Model and Scaler
# ----------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/water_quality_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ----------------------------------------
# Load Local Data (mocked for offline)
# ----------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/country_potability_scores.csv")
    if "latitude" not in df or "longitude" not in df:
        np.random.seed(42)
        df["latitude"] = np.random.uniform(8, 37, size=len(df))
        df["longitude"] = np.random.uniform(68, 97, size=len(df))
    return df

df = load_data()

# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------

st.sidebar.markdown("<h2>EXPLORE</h2>", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "🔮 Predict"

st.markdown("""
<style>
div[role="radiogroup"] {
    display: flex;
    gap: 20px;
}
div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 6px;
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ("🔮 Predict",  "📊 Dashboard", "🗺️ GeoMap", "🧪 What-if Analysis", "🧩 Quizyy"),
    index=["🔮 Predict",  "📊 Dashboard", "🗺️ GeoMap", "🧪 What-if Analysis", "🧩 Quizyy"].index(st.session_state.page),
    key="page_selector"
)
st.session_state.page = page

# ----------------------------------------
# Page: Predict
# ----------------------------------------
if page == "🔮 Predict":
    st.title("💧 Water Quality Predictor")
    st.header("🔍 Check Your Water Quality")

    def float_input(label, help_text=""):
        value = st.text_input(label, help=help_text)
        try:
            return float(value) if value else None
        except ValueError:
            st.warning(f"⚠️ Please enter a valid number for '{label}'")
            return None

    inputs = {
        "pH": float_input("pH (scale 0–14)", "Indicates acidity or basicity of water."),
        "Hardness": float_input("Hardness (mg/L)", "Calcium & magnesium content."),
        "Solids": float_input("Solids (mg/L)", "Total dissolved solids."),
        "Chloramines": float_input("Chloramines (mg/L)", "Used for water disinfection."),
        "Sulfate": float_input("Sulfate (mg/L)", "High levels can affect taste."),
        "Conductivity": float_input("Conductivity (μS/cm)", "Indicates ion concentration."),
        "Organic_carbon": float_input("Organic Carbon (mg/L)", "Presence of organic compounds."),
        "Trihalomethanes": float_input("Trihalomethanes (μg/L)", "By-product of chlorination."),
        "Turbidity": float_input("Turbidity (NTU)", "Water cloudiness.")
    }

    if st.button("🔮 Predict"):
        if None in inputs.values():
            st.warning("⚠️ Please enter all values before predicting.")
        else:
            feature_array = np.array([list(inputs.values())])
            scaled_features = scaler.transform(feature_array)
            prediction = model.predict(scaled_features)

            result_text = (
                "✅ The water is POTABLE (Safe to drink)."
                if prediction[0] == 1
                else "❌ The water is NOT POTABLE (Unsafe to drink)."
            )

            if prediction[0] == 1:
                st.success(result_text)
            else:
                st.error(result_text)


            report = "Water Quality Prediction Report\n\n"
            report += "Input Parameters:\n" + "\n".join(f"{k}: {v}" for k, v in inputs.items())
            report += f"\n\nPrediction Result:\n{result_text}"

            st.download_button(
                label="📥 Download Prediction Report",
                data=report.encode('utf-8'),
                file_name="water_quality_report.txt",
                mime="text/plain"
            )

# ----------------------------------------
# Page: Dashboard
# ----------------------------------------
elif page == "📊 Dashboard":
    st.title("📊 Water Quality Data Insights")
    st.markdown("Explore distributions and relationships in the dataset.")

    selected_feature = st.selectbox("Select a feature to visualize", df.columns[:-3])
    st.bar_chart(df[selected_feature])

    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.drop(columns=["latitude", "longitude"]).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------------------------------
# Page: GeoMap
# ----------------------------------------
elif page == "🗺️ GeoMap":
    st.subheader("🗺️ Water Quality GeoMap")
    st.markdown("Visualize potable and non-potable zones.")
    
    if "latitude" in df.columns and "longitude" in df.columns:
        m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=4)
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df.iterrows():
            color = "green" if row["Potability"] == 1 else "red"
            popup = f"""
            <b>pH:</b> {row.get('ph', 'NA')}<br>
            <b>Hardness:</b> {row.get('Hardness', 'NA')}<br>
            <b>Potable:</b> {bool(row['Potability'])}
            """
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=popup
            ).add_to(marker_cluster)

        st_folium(m, width=700, height=500)
    else:
        st.warning("Geolocation columns not found in dataset.")

# ----------------------------------------
# Page: Explainability (SHAP Summary)
# ----------------------------------------
 
if page == "🧪 What-if Analysis":
    st.title("🧪 What-if Scenario Simulator")

    # Get only the features used for training
    try:
        feature_cols = list(scaler.feature_names_in_)  # safest way
    except AttributeError:
        feature_cols = df.drop(columns=["Potability", "latitude"], errors="ignore").columns.tolist()

    # Collect user inputs via sliders
    user_input = {}
    for feature in feature_cols:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
        mean_val = float(df[feature].mean())

        user_input[feature] = st.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val,
            step=(max_val - min_val) / 100
        )

    input_df = pd.DataFrame([user_input])
    st.write("### Input values:")
    st.dataframe(input_df)

    # Make sure input_df has only the scaler/model features
    try:
        input_df = input_df[scaler.feature_names_in_]
    except AttributeError:
        pass

    # Scale the inputs
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Water is predicted to be **potable** based on your inputs.")
    else:
        st.error("❌ Water is predicted to be **not potable** based on your inputs.")
# ----------------------------------------
# Page: Quizyy
# ----------------------------------------
elif page == "🧩 Quizyy":
    st.title("🧩 Quiz Time!")
    st.markdown("Can you outsmart a drop of water? Let’s find out!")

    placeholder = "Select an option"

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    questions = {
        "quiz_q1": {
            "question": "1. What does a high pH (above 7) usually indicate?",
            "options": [
                placeholder,
                "A lemon's worst nightmare (Basic water)",
                "Water with identity issues (Neutral)",
                "The next acid rain candidate (Acidic)"
            ],
            "correct": "A lemon's worst nightmare"
        },
        "quiz_q2": {
            "question": "2. Which chemical helps kill bacteria in drinking water?",
            "options": [
                placeholder,
                "Chloramines (Bacteria's arch-nemesis)",
                "Sugar (Sweet, but deadly? Nope.)",
                "Oxygen (Vital for life, not for disinfection)"
            ],
            "correct": "Chloramines"
        },
        "quiz_q3": {
            "question": "3. What does turbidity measure?",
            "options": [
                placeholder,
                "Water's mood swings",
                "Cloudiness or how murky it looks",
                "How much glitter you accidentally spilled in it"
            ],
            "correct": "Cloudiness"
        },
        "quiz_q4": {
            "question": "4. What’s the *correct* reaction to clean, safe water?",
            "options": [
                placeholder,
                "💧😋 – *Hydration sensation!*",
                "😬 – *Tastes like disappointment*",
                "💀 – *Definitely not recommended*"
            ],
            "correct": "💧😋"
        }
    }

    answers = {}
    for key, q in questions.items():
        answers[key] = st.radio(q["question"], q["options"], index=0, key=key)

    if st.button("🎯 Submit Quiz"):
        if placeholder in answers.values():
            st.warning("⚠️ Please answer all questions before submitting.")
        else:
            score = 0
            for key, q in questions.items():
                if answers[key].startswith(q["correct"]):
                    score += 1
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True

    if st.session_state.quiz_submitted:
        score = st.session_state.quiz_score
        st.success(f"🎉 You scored {score}/4!")

        if score == 4:
            st.balloons()
            st.markdown("💎 You're a certified **H2-OMG** genius! 🧠💧")
        elif score == 3:
            st.markdown("👏 Nice job! You clearly hydrate AND educate.")
        elif score == 2:
            st.markdown("😌 Decent effort. Maybe skim the water safety manual again?")
        else:
            st.markdown("🧪 Uh oh... did you drink the quiz water? Time for a refresher! 😅")

        if st.button("🔁 Retake Quiz"):
            # Clear all quiz-related state values
            for key in list(st.session_state.keys()):
                if key.startswith("quiz_q") or key in ["quiz_submitted", "quiz_score"]:
                    del st.session_state[key]
            st.rerun()

