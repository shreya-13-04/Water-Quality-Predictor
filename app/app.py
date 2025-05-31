import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import numpy as np
import pydeck as pdk
import json

# Streamlit Page Setup
st.set_page_config(page_title="Water Quality Predictor", page_icon="💧", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("models/water_quality_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("data/water_potability.csv")
    np.random.seed(42)
    df["latitude"] = np.random.uniform(8, 37, size=len(df))   # India's lat range
    df["longitude"] = np.random.uniform(68, 97, size=len(df)) # India's lon range
    return df

df = load_data()

st.sidebar.markdown("<h2>EXPLORE</h2>", unsafe_allow_html=True)


# Initialize session state for page
if "page" not in st.session_state:
    st.session_state.page = "🔮 Predict"  # default page

st.markdown("""
<style>
/* Make radio buttons horizontal */
div[role="radiogroup"] {
    display: flex;
    gap: 20px;  /* space between options */
}

div[role="radiogroup"] > label {
    display: flex;
    align-items: center;
    gap: 6px;  /* space between circle and text */
}
</style>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "",
    ("🔮 Predict", "📊 Visualize Data", "🗺️ Map", "🧩 Quizyy"),
    index=["🔮 Predict", "📊 Visualize Data", "🗺️ Map", "🧩 Quizyy"].index(st.session_state.page),
    key="page_selector"
)

# Update session state page
st.session_state.page = page



# --------------------------------------------
# 🔮 Page 1: Water Quality Prediction
# --------------------------------------------
if page == "🔮 Predict":
    st.title("💧 Water Quality Predictor")
    st.header("🔍 Check Your Water Quality")

    # Input fields
    def float_input(label):
        value = st.text_input(label)
        try:
            return float(value) if value else None
        except ValueError:
            st.warning(f"⚠️ Please enter a valid number for '{label}'")
            return None

    ph = float_input("pH (scale 0-14)")
    Hardness = float_input("Hardness (mg/L)")
    Solids = float_input("Solids (mg/L)")
    Chloramines = float_input("Chloramines (mg/L)")
    Sulfate = float_input("Sulfate (mg/L)")
    Conductivity = float_input("Conductivity (μS/cm)")
    Organic_carbon = float_input("Organic Carbon (mg/L)")
    Trihalomethanes = float_input("Trihalomethanes (μg/L)")
    Turbidity = float_input("Turbidity (NTU)")
    if st.button("🔮 Predict"):
        features = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
                              Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)

        if prediction[0] == 1:
            result_text = "✅ The water is POTABLE (Safe to drink)."
            st.success(result_text)
        else:
            result_text = "❌ The water is NOT POTABLE (Unsafe to drink)."
            st.error(result_text)

        report = f"""
Water Quality Prediction Report

Input Parameters:
-----------------
pH: {ph}
Hardness: {Hardness}
Solids: {Solids}
Chloramines: {Chloramines}
Sulfate: {Sulfate}
Conductivity: {Conductivity}
Organic Carbon: {Organic_carbon}
Trihalomethanes: {Trihalomethanes}
Turbidity: {Turbidity}

Prediction Result:
------------------
{result_text}
"""
        st.download_button(
            label="📥 Download Prediction Report",
            data=report.encode('utf-8'),
            file_name="water_quality_report.txt",
            mime="text/plain"
        )

# --------------------------------------------
# 📊 Page 2: Data Visualization
# --------------------------------------------
elif page == "📊 Visualize Data":
    st.title("📊 Visual Analysis of Water Quality")

    st.subheader("🔬 pH Distribution by Potability")
    fig1, ax1 = plt.subplots()
    sns.histplot(data=df, x="ph", hue="Potability", kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📦 Boxplot Comparison (pH vs Potability)")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="Potability", y="ph", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("🧠 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# --------------------------------------------
# 🗺️ Page 3: Geospatial Water Potability Map
# --------------------------------------------
elif page == "🗺️ Map":
    st.title("🌍 Country-level Water Potability")
    st.markdown("### Potability Levels: Green = High, Red = Low")

    try:
        with open("data/custom.geo.json", encoding="utf-8") as f:
            geojson = json.load(f)
    except FileNotFoundError:
        st.error("❌ GeoJSON file not found. Please ensure 'data/custom.geo.json' exists.")
        st.stop()

    real_scores = {
        "India": 0.62, "United States": 0.91, "Nigeria": 0.33,
        "Brazil": 0.75, "Canada": 0.95, "Russia": 0.88,
        "China": 0.60, "Australia": 0.93, "South Africa": 0.51,
        "Pakistan": 0.45
    }

    for feature in geojson["features"]:
        name = feature["properties"].get("NAME") or feature["properties"].get("name")
        score = real_scores.get(name, round(np.random.uniform(0.3, 0.9), 2))
        feature["properties"]["potability_score"] = score

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        opacity=0.7,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="""
            [
                255 * (1 - properties.potability_score),
                255 * properties.potability_score,
                80
            ]
        """,
        get_line_color=[40, 40, 40],
        line_width_min_pixels=1,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=20,
        longitude=0,
        zoom=1.2,
        pitch=0
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "Potability Score: <b>{potability_score}</b>",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))
# --------------------------------------------
# 🧩 Page 4: Quiz Placeholder
# --------------------------------------------
elif page == "🧩 Quizyy":
    st.markdown("""
        <style>
        .quiz-container * {
            font-size: 40px !important;
        }
        </style>
        <div class="quiz-container">
    """, unsafe_allow_html=True)
    st.title("🧩 Quiz Time!")
    st.markdown("Can you outsmart a drop of water? Let's see!")

    # Initialize state
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0

    # Placeholder option for blank default
    placeholder = "Select an option"

    # Questions with blank default
    q1 = st.radio(
        "1. What does a high pH (above 7) usually indicate?",
        [placeholder, 
         "A lemon's worst nightmare (Basic water)",
         "Water with identity issues (Neutral)",
         "The next acid rain candidate (Acidic)"],
        index=0,
        key="quiz_q1"
    )

    q2 = st.radio(
        "2. Which chemical helps kill bacteria in drinking water?",
        [placeholder,
         "Chloramines (Bacteria's arch-nemesis)",
         "Sugar (Sweet, but deadly? Nope.)",
         "Oxygen (Vital for life, not for disinfection)"],
        index=0,
        key="quiz_q2"
    )

    q3 = st.radio(
        "3. What does turbidity measure?",
        [placeholder,
         "Water's mood swings",
         "Cloudiness or how murky it looks",
         "How much glitter you accidentally spilled in it"],
        index=0,
        key="quiz_q3"
    )

    q4 = st.radio(
        "4. What’s the *correct* reaction to clean, safe water?",
        [placeholder,
         "💧😋 – *Hydration sensation!*",
         "😬 – *Tastes like disappointment*",
         "💀 – *Definitely not recommended*"],
        index=0,
        key="quiz_q4"
    )

    # Submit Button
    if st.button("🎯 Submit Quiz"):
        # Ensure all questions are answered
        if (
            st.session_state.quiz_q1 == placeholder or
            st.session_state.quiz_q2 == placeholder or
            st.session_state.quiz_q3 == placeholder or
            st.session_state.quiz_q4 == placeholder
        ):
            st.warning("⚠️ Please answer all questions before submitting.")
        else:
            score = 0
            if st.session_state.quiz_q1.startswith("A lemon's worst nightmare"):
                score += 1
            if st.session_state.quiz_q2.startswith("Chloramines"):
                score += 1
            if "Cloudiness" in st.session_state.quiz_q3:
                score += 1
            if st.session_state.quiz_q4.startswith("💧😋"):
                score += 1

            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True

    # Display Results
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
