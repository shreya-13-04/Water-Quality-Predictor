
# 💧 Water Quality Predictor

An interactive AI-powered Streamlit web app to predict whether water is **safe** or **unsafe** to drink based on various physicochemical parameters.

## 🌐 Live Demo

*Coming soon* (You can deploy on [Streamlit Cloud](https://streamlit.io/cloud) or [Render](https://render.com/)).

---

## 🚀 Features

- ✅ Predicts potability (safe/unsafe) using machine learning
- 🧪 **What-if Analysis** with sliders for parameter tuning
- 📊 Visualization Dashboard for trends and correlation insights
- 🗺️ Interactive Geospatial Mapping with radius filtering
- 🧩 Engaging Water Awareness Quiz
- 📱 User-friendly, responsive UI
- 🧠 Based on real-world water quality dataset

---

## 📂 Project Structure

```
📁 water-quality-predictor/
│
├── app                      # Main Streamlit app
    ├── app.py
├── data                     # Original dataset  
    ├──custom.geo.json
    ├──water_potability.csv
├── models                   #Trained ML model
    ├──scaler.pkl
    ├──water_quality_model.pkl
├── notebooks
    ├──eda.ipynb
    ├──preprocessing_modeling.ipynb
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🧪 Parameters and Units

| Parameter           | Unit                            |
|---------------------|----------------------------------|
| pH                  | Unitless (scale 0–14)            |
| Hardness            | mg/L (as CaCO₃)                  |
| Solids              | mg/L                             |
| Chloramines         | mg/L                             |
| Sulfate             | mg/L                             |
| Conductivity        | μS/cm (microsiemens/cm)          |
| Organic Carbon      | mg/L                             |
| Trihalomethanes     | µg/L                             |
| Turbidity           | NTU (Nephelometric Turbidity Unit) |

---

## 💡 Sample Inputs

### ✅ Safe Water
```
pH = 7.2  
Hardness = 150  
Solids = 500  
Chloramines = 4  
Sulfate = 250  
Conductivity = 400  
Organic Carbon = 9  
Trihalomethanes = 50  
Turbidity = 3  
```

### ❌ Unsafe Water
```
pH = 12.5 
Hardness = 400  
Solids = 16000  
Chloramines = 12  
Sulfate = 500  
Conductivity = 1000  
Organic Carbon = 30  
Trihalomethanes = 130  
Turbidity = 10  
```

---

## 📦 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/shreya-13-04/water-quality-predictor.git
   cd water-quality-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   python -m streamlit run app/app.py
   ```

---

## 📈 Model

A machine learning classification model (e.g., Random Forest or Logistic Regression) is trained on the [Water Potability dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability). The model predicts the **potability (1 or 0)** based on the given inputs.

---

## 🧠 Quiz Feature

Challenge users to test their water safety knowledge using the fun and educational **Quizyy** section with:

- Multiple-choice questions
- Score tracking
- Custom feedback

---

## ✨ Future Enhancements

📌 Future Enhancements

-🔍 SHAP / LIME-based explainability
-🌐 Real-time water quality API integration
-📦 Dockerize for production deployment

---

## 🙋‍♀️ Author

Made with 💙 by **Shreya B**

> A project exploring the intersection of machine learning, environmental health, and user interaction.

---

## 📄 License

MIT License. Feel free to fork, modify, and share with attribution.
