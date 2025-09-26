# %%
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# App Title
# -----------------------------
st.title("Glaucoma Prediction App")

# -----------------------------
# Load trained model & scaler
# -----------------------------
try:
    model = joblib.load("glaucoma_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'glaucoma_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Patient Details")

# Numerical features
age = st.sidebar.slider("Age", min_value=20, max_value=90, value=50)
iop = st.sidebar.slider("Intraocular Pressure (IOP)", min_value=8.0, max_value=40.0, value=15.0, step=0.1)
cdr = st.sidebar.slider("Cup-to-Disc Ratio (CDR)", min_value=0.1, max_value=1.0, value=0.5, step=0.01)
pachy = st.sidebar.slider("Pachymetry (μm)", min_value=400.0, max_value=650.0, value=520.0, step=1.0)

# Categorical features
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
family_history = st.sidebar.selectbox("Family History", ["Yes", "No"])
cataract_status = st.sidebar.selectbox("Cataract Status", ["Present", "Absent"])
angle_closure = st.sidebar.selectbox("Angle Closure Status", ["Open", "Closed"])
visual_symptoms = st.sidebar.selectbox("Visual Symptoms", ["Blurred Vision", "Eye Pain", "Headache", "None"])
visual_acuity = st.sidebar.selectbox("Visual Acuity Measurements", ["Normal", "Reduced"])
visual_field = st.sidebar.selectbox("Visual Field Test Results", ["Normal", "Defective"])
oct = st.sidebar.selectbox("OCT Results", ["Normal", "Abnormal"])

# -----------------------------
# Preprocess Input
# -----------------------------
def preprocess_input(age, iop, cdr, pachy, gender, family_history,
                     cataract_status, angle_closure, visual_symptoms,
                     visual_acuity, visual_field, oct):
    
    data = {
        "Age": age,
        "Intraocular Pressure (IOP)": iop,
        "Cup-to-Disc Ratio (CDR)": cdr,
        "Pachymetry": pachy,
        "Gender": 1 if gender == "Male" else 0,
        "Family History": 1 if family_history == "Yes" else 0,
        "Cataract Status": 1 if cataract_status == "Present" else 0,
        "Angle Closure Status": 1 if angle_closure == "Closed" else 0,
        "Visual Symptoms": 0 if visual_symptoms == "None" else 1,
        "Visual Acuity Measurements": 0 if visual_acuity == "Normal" else 1,
        "Visual Field Test Results": 0 if visual_field == "Normal" else 1,
        "Optical Coherence Tomography (OCT) Results": 0 if oct == "Normal" else 1
    }
    
    df = pd.DataFrame([data])

    # Scale numeric features
    num_cols = ["Age", "Intraocular Pressure (IOP)", "Cup-to-Disc Ratio (CDR)", "Pachymetry"]
    df[num_cols] = scaler.transform(df[num_cols])
    
    return df

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict"):
    try:
        input_df = preprocess_input(
            age, iop, cdr, pachy, gender, family_history,
            cataract_status, angle_closure, visual_symptoms,
            visual_acuity, visual_field, oct
        )

        prediction = model.predict(input_df)[0]

        st.subheader("Prediction Result")
        st.write(f"**Diagnosis Prediction:** {prediction}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

# -----------------------------
# Instructions
# -----------------------------
st.write("""
### Instructions
1. Enter patient details in the sidebar.
2. Fill in both numerical and categorical fields.
3. Click **Predict** to see the glaucoma diagnosis/type.
""")



