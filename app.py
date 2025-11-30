import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("water_quality_model.pkl")

st.title("🌊 Water Quality Classification (Region 10)")
st.write("This tool predicts whether a waterbody is **Safe** or **Not Safe** based on pH and Fecal Coliform values.")

# Inputs
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
fc = st.number_input("Fecal Coliform (MPN/100 mL)", min_value=0, max_value=2000, value=100)

if st.button("Predict"):
    sample = pd.DataFrame([[pH, fc]], columns=["pH","FecalColiform"])
    pred = model.predict(sample)[0]

    if pred == 1:
        st.success("Result: SAFE ✔️")
    else:
        st.error("Result: NOT SAFE ❌")
