import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ======================
# Page config
# ======================
st.set_page_config(
    page_title="Anemia Disease Prediction",
    page_icon="🩸",
    layout="centered"
)

st.title("🩸 Anemia Disease Prediction System")
st.write("Predict anemia type using blood test parameters")

# ======================
# Load model & encoder
# ======================
@st.cache_resource
def load_model():
    model = joblib.load("/content/anemia_pipeline.pkl")
    le = joblib.load("/content/label_encoder.pkl")
    return model, le

model, le = load_model()

# ======================
# Sidebar
# ======================
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose input type:",
    ("Manual Entry", "Upload CSV")
)

# ======================
# Manual Input
# ======================
if input_method == "Manual Entry":
    st.subheader("🧪 Enter Patient Details")

    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])

    rbc = st.number_input("RBC (10¹²/L)", value=4.5)
    hgb = st.number_input("HGB (g/dL)", value=13.5)
    hct = st.number_input("HCT (%)", value=40.0)
    mcv = st.number_input("MCV (fL)", value=90.0)
    mch = st.number_input("MCH (pg)", value=30.0)
    mchc = st.number_input("MCHC (g/dL)", value=33.0)
    rdw = st.number_input("RDW-CV (%)", value=13.0)

    if st.button("🔍 Predict"):
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'RBC10¹²-L': rbc,
            'HGBg-dL': hgb,
            'HCT%': hct,
            'MCVfL': mcv,
            'MCHpg': mch,
            'MCHCg-dL': mchc,
            'RDW-CV%': rdw
        }])

        probs = model.predict_proba(input_df)[0]
        pred_idx = np.argmax(probs)

        disease = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx] * 100

        st.success(f"🧬 Predicted Disease: **{disease}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")

        if confidence < 60:
            st.warning("⚠️ Low confidence prediction. Further clinical evaluation is recommended.")

# ======================
# CSV Upload
# ======================
else:
    st.subheader("📂 Upload CSV File")
    st.write("CSV must contain the same feature columns used during training.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        predictions = model.predict(df)
        probs = model.predict_proba(df)

        df["Predicted_Disease"] = le.inverse_transform(predictions)
        df["Confidence (%)"] = np.max(probs, axis=1) * 100

        st.success("✅ Prediction completed")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results",
            data=csv,
            file_name="anemia_predictions.csv",
            mime="text/csv"
        )

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("⚕️ This tool is for academic & research purposes only.")
