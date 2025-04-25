import streamlit as st
import pandas as pd
import joblib
import base64

# Set page config first
st.set_page_config(page_title="SIM Deactivation Predictor", page_icon="📱")

# ---- Embed background image using base64 with gradient ----
def add_bg_with_gradient(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)),
                    url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}

    /* Responsive tweaks */
    @media (max-width: 768px) {{
        .stSlider, .stNumberInput, .stSelectbox, .stRadio, .stButton {{
            font-size: 16px !important;
        }}
        .stMarkdown h1, .stMarkdown h2 {{
            font-size: 24px !important;
        }}
        .stForm {{
            padding: 1rem !important;
        }}
    }}

    .main > div {{
        background-color: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
    }}
    </style>
    """, unsafe_allow_html=True)

add_bg_with_gradient("background.jpg")  # Make sure this image is in the same folder as app.py

# ---- Load model and encoders ----
try:
    model = joblib.load("sim_churn_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or encoders: {e}")
    st.stop()

# ---- App Title ----
st.title("📱 SIM Deactivation Prediction App")
st.markdown("Predict whether a SIM is likely to be deactivated based on recent usage data.")

# ---- Sidebar Input Form ----
st.sidebar.header("Enter SIM Data for Prediction")

total_calls = st.sidebar.slider("📞 Total Calls (Last 30 Days)", 0, 200, 50)
total_sms = st.sidebar.slider("💬 Total SMS (Last 30 Days)", 0, 100, 10)
total_data_mb = st.sidebar.slider("🌐 Total Data Used (MB)", 0, 10000, 3000)
avg_daily_usage_mb = st.sidebar.number_input("📊 Average Daily Usage (MB)", value=100.0)
inactivity_days = st.sidebar.selectbox("🛌 Days of Inactivity", [0, 1, 5, 10, 15, 30, 45])
sim_age_days = st.sidebar.slider("📆 SIM Age (Days)", 30, 730, 365)
is_prepaid = st.sidebar.radio("💳 SIM Type", ["Prepaid", "Postpaid"])
region = st.sidebar.selectbox("📍 Region", ["North", "South", "East", "West", "North-East"])
support_calls = st.sidebar.slider("🆘 Support Calls", 0, 10, 0)

# Form submission button
submitted = st.sidebar.button("🔍 Submit")

# ---- Prediction ----
if submitted:
    input_df = pd.DataFrame([{
        "total_calls": total_calls,
        "total_sms": total_sms,
        "total_data_mb": total_data_mb,
        "avg_daily_usage_mb": avg_daily_usage_mb,
        "inactivity_days": inactivity_days,
        "sim_age_days": sim_age_days,
        "is_prepaid": 1 if is_prepaid == "Prepaid" else 0,
        "region": region,
        "support_calls": support_calls
    }])

    if "region" in label_encoders:
        input_df["region"] = label_encoders["region"].transform(input_df["region"])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # ---- Display Prediction Result and Insights ----
    st.subheader("📌 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **SIM is likely to be deactivated**! (Probability: {probability*100:.1f}%)")
        st.write("### **Insights**:")
        st.write("This SIM shows signs of being inactive and has a higher chance of deactivation. Consider the following:")
        st.write("- **Days of Inactivity:** SIM has been inactive for a while.")
        st.write("- **Low usage metrics:** Low data, calls, and SMS may indicate declining usage.")
        st.write("- **Support Calls:** Multiple support calls can be a signal of dissatisfaction.")
    else:
        st.success(f"✅ **SIM is likely to remain active**. (Deactivation probability: {probability*100:.1f}%)")
        st.write("### **Insights**:")
        st.write("This SIM has a healthy usage profile and is unlikely to be deactivated.")
        st.write("- **High usage metrics:** Consistent data, calls, and SMS usage.")
        st.write("- **Recent activity:** No prolonged inactivity detected.")
        st.write("- **Low support calls:** Indicates fewer issues with the service.")
