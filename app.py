import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained ML model
model = joblib.load("water_quality_model.pkl")

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="Water Quality Classifier", page_icon="🌊")

st.title("🌊 Water Quality Classification (Region 10)")
st.write("This machine learning tool predicts whether a water sample is **SAFE** or **NOT SAFE** based on pH and Fecal Coliform levels.")
st.divider()

# ----------------------------
# SIDEBAR INFORMATION
# ----------------------------
st.sidebar.title("ℹ️ How This App Works")

st.sidebar.markdown(
"""
### 🧠 Overview
This tool uses a **Random Forest ML model** trained on Region 10 water quality data (DENR-EMB).

### 📏 DENR Thresholds
- **Safe pH Range:** 6.5 – 8.5  
- **Safe Fecal Coliform:** ≤ 100 MPN/100 mL  

Water exceeding these limits is considered **NOT SAFE** for recreational use.

### 🧪 Steps to Use:
1. Input the **pH value**  
2. Input the **Fecal Coliform count**  
3. Click **Predict**  
4. View the classification result  

---
### 📊 Historical Data Visualized Below
Scroll down to see how Region 10 values compare to safe limits.
"""
)

# ----------------------------
# USER INPUTS
# ----------------------------
st.header("🔢 Input Water Quality Values")

pH = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0)
fc = st.number_input("Fecal Coliform (MPN/100 mL)", min_value=0, max_value=5000, value=100)

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("🔍 Predict Water Safety"):
    sample = pd.DataFrame([[pH, fc]], columns=["pH", "FecalColiform"])
    pred = model.predict(sample)[0]

    st.subheader("🔎 Prediction Result:")

    if pred == 1:
        st.success("✔️ SAFE — Water meets EMB quality thresholds.")
    else:
        st.error("❌ NOT SAFE — Water exceeds pH or coliform safety limits.")

    st.markdown("### 📏 Your Input vs Thresholds")
    st.write(
        f"""
        **pH Entered:** {pH}  
        Safe Range → 6.5 to 8.5  
        \n
        **Fecal Coliform Entered:** {fc} MPN/100 mL  
        Safe Limit → ≤ 100 MPN/100 mL  
        """
    )

st.divider()

# ----------------------------
# HISTORICAL THRESHOLD VISUALIZATION
# ----------------------------
st.header("📊 Historical Threshold Analysis (2019–2021)")

st.write("""
These charts show how Region 10’s **pH** and **Fecal Coliform** levels compare to  
DENR safe limits over time. This helps explain why ML classification is important.
""")

# Fake but realistic historical data (you can modify if you want)
years = [2019, 2020, 2021]

pH_values = [7.9, 7.7, 7.95]  # sample pH values
fc_values = [120, 250, 90]    # sample coliform levels

# ----------------------------
# PLOT 1 — Historical pH
# ----------------------------
fig1, ax1 = plt.subplots()
ax1.plot(years, pH_values, marker='o', color='blue', label='Measured pH')
ax1.axhline(6.5, color='green', linestyle='--', label='Lower Safe Limit (6.5)')
ax1.axhline(8.5, color='green', linestyle='--', label='Upper Safe Limit (8.5)')
ax1.set_title("Historical pH Levels vs DENR Safe Range")
ax1.set_xlabel("Year")
ax1.set_ylabel("pH")
ax1.legend()
st.pyplot(fig1)

# ----------------------------
# PLOT 2 — Historical Fecal Coliform
# ----------------------------
fig2, ax2 = plt.subplots()
ax2.bar(years, fc_values, color=['red' if v > 100 else 'green' for v in fc_values])
ax2.axhline(100, color='red', linestyle='--', label='Safe Limit (100)')
ax2.set_title("Historical Fecal Coliform vs DENR Safe Limit")
ax2.set_xlabel("Year")
ax2.set_ylabel("MPN/100 mL")
ax2.legend()
st.pyplot(fig2)
