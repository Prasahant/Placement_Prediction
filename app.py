import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Placement Prediction", layout="wide")

# ---------------- LOAD MODEL ----------------
# Make sure you save your trained model as 'placement_model.pkl'
@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")

model = load_model()

# ---------------- TITLE ----------------
st.title("🎓 Placement Classification Dashboard")
st.markdown("Predict whether a student will be **Placed or Not Placed** based on model trained")

# ---------------- INPUT SECTION ----------------
st.subheader("📝 Student Details")

col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.1)

with col2:
    iq = st.number_input("IQ", min_value=30, max_value=200, step=1)

# ---------------- PREDICTION ----------------
st.subheader("🔮 Prediction")

if st.button("Predict Placement"):
    input_data = np.array([[cgpa, iq]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Student is likely to be PLACED")
    else:
        st.error("❌ Student is NOT likely to be placed")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("ML Frontend built with Streamlit | Placement Classification")
