import streamlit as st
import pandas as pd
import joblib

# Load trained ML model
model = joblib.load("water_quality_model.pkl")

# ----------------------------
# Page UI
# ----------------------------
st.set_page_config(page_title="Water Quality Classifier", page_icon="🌊")

st.title("🌊 Water Quality Classification (Region 10)")
st.write("This machine learning tool predicts whether a water sample is **SAFE** or **NOT SAFE** based on pH and Fecal Coliform levels.")

st.divider()

# ----------------------------
# Sidebar – Info + Instructions
# ----------------------------
st.sidebar.title("ℹ️ About This App")

st.sidebar.markdown(
"""
This tool uses a **Random Forest Machine Learning Model** trained on **Region 10 water quality data** 
from the **DENR – Environmental Management Bureau (EMB)**.

### 📌 Classification Thresholds (Based on DENR Standards)
- **pH Safe Range:** *6.5 to 8.5*
- **Fecal Coliform Safe Limit:** *≤ 100 MPN/100 mL*  
  (Above 100 is considered **unsafe** for recreational waters)

### 🧪 How It Works
1. Enter the **pH** value  
2. Enter the **Fecal Coliform** count  
3. Click **Predict**  
4. The ML model classifies the water as:
   - ✔️ **SAFE**  
   - ❌ **NOT SAFE**

### 🧠 Notes
- The model learns patterns from Region 10 dataset  
- This is for **educational/demo** purposes  
"""
)

# ----------------------------
# Input Fields
# ----------------------------
st.header("🔢 Input Water Quality Values")

pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)

fc = st.number_input(
    "Fecal Coliform (MPN/100 mL)",
    min_value=0,
    max_value=5000,
    value=100
)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Predict Water Safety"):
    sample = pd.DataFrame([[pH, fc]], columns=["pH", "FecalColiform"])
    pred = model.predict(sample)[0]

    st.subheader("🔎 Prediction Result:")

    if pred == 1:
        st.success("✔️ SAFE — Water meets EMB water quality thresholds.")
    else:
        st.error("❌ NOT SAFE — Water exceeds safe coliform limits or pH is outside the safe range.")

    # Display criteria used
    st.markdown("### 📏 Evaluation Based on Standards")
    st.write(
        f"""
        - Entered **pH**: {pH}  
          (Safe range: **6.5–8.5**)  
        - Entered **Fecal Coliform**: {fc} MPN/100 mL  
          (Safe limit: **≤ 100 MPN/100 mL**)
        """
    )

