import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("water_quality_model.pkl")

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Water Quality Classifier", page_icon="🌊")

st.title("🌊 Water Quality Classification (Region 10)")
st.write("This ML tool predicts whether water is **SAFE** or **NOT SAFE** using 6 environmental parameters.")
st.divider()

# ----------------------------
# SIDEBAR INFORMATION
# ----------------------------
st.sidebar.title("ℹ️ Model Information")

# Update this after training
ACCURACY = 0.95  
st.sidebar.success(f"Model Accuracy: {ACCURACY * 100:.2f}%")

st.sidebar.markdown(
"""
### 📏 DENR Thresholds

**pH:** 6.5 – 8.5  
**Fecal Coliform:** ≤ 100 MPN/100 mL  
**Dissolved Oxygen:** ≥ 5 mg/L  
**BOD:** ≤ 5 mg/L  
**Turbidity:** ≤ 5 NTU  
**Temperature:** ≤ 30°C  

---
### 🧪 How To Use
1. Enter all water-quality parameters  
2. Click **Predict Water Safety**  
3. View classification result and parameter analysis  

---
### 🧠 Notes
- Trained using expanded Region 10-style dataset  
- Random Forest (300 trees)  
"""
)

# ----------------------------
# USER INPUT FIELDS (6 PARAMETERS)
# ----------------------------
st.header("🔢 Enter Water Quality Parameters")

col1, col2 = st.columns(2)

with col1:
    pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
    DO = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=6.0)
    Turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=50.0, value=3.0)

with col2:
    FecalColiform = st.number_input("Fecal Coliform (MPN/100 mL)", min_value=0, max_value=5000, value=100)
    BOD = st.number_input("Biochemical Oxygen Demand (mg/L)", min_value=0.0, max_value=20.0, value=3.0)
    Temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=40.0, value=28.0)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🔍 Predict Water Safety"):
    
    sample = pd.DataFrame([[
        pH, FecalColiform, DO, BOD, Turbidity, Temp
    ]], columns=["pH", "FecalColiform", "DO", "BOD", "Turbidity", "Temp"])

    pred = model.predict(sample)[0]

    st.subheader("🔎 Prediction Result:")

    if pred == 1:
        st.success("✔️ SAFE — Water meets EMB safety thresholds.")
    else:
        st.error("❌ NOT SAFE — One or more parameters exceed safe limits.")

    # Detailed parameter analysis
    st.markdown("### 📊 Parameter Evaluation")

    st.write(f"**pH:** {pH} (Safe: 6.5–8.5)")
    st.write(f"**Fecal Coliform:** {FecalColiform} (Safe: ≤ 100)")
    st.write(f"**Dissolved Oxygen:** {DO} mg/L (Safe: ≥ 5)")
    st.write(f"**BOD:** {BOD} mg/L (Safe: ≤ 5)")
    st.write(f"**Turbidity:** {Turbidity} NTU (Safe: ≤ 5)")
    st.write(f"**Temperature:** {Temp} °C (Safe: ≤ 30)")

