# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(page_title="Collision Estimator", layout="centered")
st.title("Εκτιμητής Κρούσης — Force, Collision Time, Δp, Δv")

# --- Helper: load data ---
@st.cache_data(show_spinner=False)
def load_data(path="Experiment.xlsx"):
    df = pd.read_excel(path)
    return df

# --- Load pre-trained models and encoder ---
@st.cache_resource(show_spinner=True)
def load_models(df):
    # rename columns
    df = df.rename(columns={
        "Υλικό Προφυλακτήρα": "Material",
        "Ύψος Εκκίνησης (m)": "Height (m)",
        "Μάζα (kg)": "Mass (kg)",
        "Πάχος Υλικού (mm)": "Material Thickness (mm)",
        "Συντελεστής Ελαστικότητας": "Elasticity",
        "Ταχύτητα (m/s)": "Velocity (m/s)",
        "Ασκούμενη Δύναμη (N)": "Force (N)",
        "Χρόνος Κρούσης (s)": "Collision Time (s)"
    })

    # Load models and encoder
    with open("model_force.pkl", "rb") as f:
        model_force = pickle.load(f)
    with open("model_time.pkl", "rb") as f:
        model_time = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return df, encoder, model_force, model_time

# Load dataset
try:
    df = load_data("Experiment.xlsx")
except Exception as e:
    st.error(f"Failed to load Experiment.xlsx. Error: {e}")
    st.stop()

# Sidebar raw data preview
st.sidebar.header("Dataset")
if st.sidebar.checkbox("Show raw data"):
    st.dataframe(df.head())

# Load models
with st.spinner("Loading pre-trained models ..."):
    df_proc, encoder, model_force, model_time = load_models(df)
st.success("Models ready — you can run predictions below.")

# --- UI controls ---
st.header("Παράμετροι εισόδου")

material_options = encoder.classes_.tolist()
material = st.selectbox("Υλικό:", material_options, index=0)

height = st.slider("Ύψος (m):", min_value=0.1, max_value=1.0, value=0.57, step=0.01)
mass = st.slider("Μάζα (kg):", min_value=0.4, max_value=1.2, value=0.65, step=0.01)
thickness = st.slider("Πάχος Υλικού (mm):", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
velocity = st.slider("Ταχύτητα (m/s):", min_value=1.0, max_value=5.0, value=3.90, step=0.1)

# Elasticity map
elasticity_map = {
    "Σκληρό": 0.3,
    "Φελλός": 0.6,
    "Ελαστικό": 1.2
}
elasticity = elasticity_map.get(material, 0.5)

# --- Prediction button ---
if st.button("Υπολογισμός"):
    try:
        material_encoded = int(encoder.transform([material])[0])
    except Exception:
        st.error("Failed to encode selected material. Check dataset's Material column.")
        st.stop()

    features = pd.DataFrame([{
        "Material_encoded": material_encoded,
        "Height (m)": height,
        "Mass (kg)": mass,
        "Material Thickness (mm)": thickness,
        "Elasticity": elasticity,
        "Velocity (m/s)": velocity
    }])

    # Predict
    pred_force = float(model_force.predict(features)[0])
    pred_time = float(model_time.predict(features)[0])

    delta_p = pred_force * pred_time
    delta_v = delta_p / mass if mass != 0 else np.nan

    # Show results
    st.subheader("Αποτελέσματα Πρόβλεψης")
    st.write(f"**Υλικό:** {material}")
    st.write(f"**Συντελεστής Ελαστικότητας (χρησιμοποιήθηκε):** {elasticity:.2f}")
    st.write(f"**Εκτιμώμενη Δύναμη:** {pred_force:.3f} N")
    st.write(f"**Εκτιμώμενος Χρόνος Κρούσης:** {pred_time:.3f} s")
    st.write(f"**Μεταβολή Ορμής (Δp):** {delta_p:.3f} N·s")
    st.write(f"**Μεταβολή Ταχύτητας (Δv):** {delta_v:.3f} m/s")

# --- Correlation heatmap ---
st.header("Heatmap Συσχετίσεων (Correlation Heatmap)")
fig, ax = plt.subplots(figsize=(8,6))
corr = df_proc.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap των Μεταβλητών")
st.pyplot(fig)
