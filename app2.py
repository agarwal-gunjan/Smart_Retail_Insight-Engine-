#python3 -m streamlit run app2.py

import streamlit as st
import pandas as pd
import pickle
import datetime

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Retail Insight Engine",
    page_icon="🛒",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
with open("model_1.pkl", "rb") as f:
    rev_1 = pickle.load(f)

with open("model_2_capped.pkl", "rb") as f:
    rev_2 = pickle.load(f)

with open("sales_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

# ---------------- HERO SECTION ----------------
st.title("🛒 Smart Retail Insight Engine")
st.markdown("### 📊 Predict • Analyze • Decide smarter")
st.progress(100)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("🚀 Project Overview")
st.sidebar.success(
    "This system predicts:\n\n"
    "💰 Daily Revenue\n"
    "📈 Sales Category\n\n"
    "**Models:** Random Forest\n\n"
    "Built for hackathons & demos"
)

# ---------------- MAIN LAYOUT ----------------
left, right = st.columns([1.2, 1])

# ================= LEFT PANEL =================
with left:

    with st.container(border=True):
        st.subheader("📅 Date Inputs")

        c1, c2, c3 = st.columns(3)
        with c1:
            day = st.number_input("Day", 1, 31)
        with c2:
            month = st.number_input("Month", 1, 12)
        with c3:
            year = st.number_input("Year", 2000, 2030)

        try:
            date_obj = datetime.date(int(year), int(month), int(day))
            dayofweek = date_obj.weekday()
            is_weekend = 1 if dayofweek >= 5 else 0

            st.info(
                f"🗓 **Day of Week:** {dayofweek}  \n"
                f"🌴 **Weekend:** {'Yes' if is_weekend else 'No'}"
            )
        except:
            st.error("❌ Invalid date")

    with st.container(border=True):
        st.subheader("⏪ Revenue History")

        revenue_lag_1 = st.number_input(
            "Revenue – 1 Day Ago",
            min_value=0.0,
            step=100.0
        )

        revenue_lag_7 = st.number_input(
            "Revenue – 7 Days Ago",
            min_value=0.0,
            step=100.0
        )

# ================= RIGHT PANEL =================
with right:

    with st.container(border=True):
        st.subheader("🔮 Predictions Dashboard")

        input_data = pd.DataFrame({
            "day": [day],
            "month": [month],
            "year": [year],
            "dayofweek": [dayofweek],
            "is_weekend": [is_weekend],
            "revenue_lag_1": [revenue_lag_1],
            "revenue_lag_7": [revenue_lag_7]
        })

        tab1, tab2, tab3 = st.tabs(
            ["💰 Revenue Model 1", "💰 Capped Revenue", "📈 Sales Category"]
        )

        with tab1:
            if st.button("Predict Revenue"):
                pred = rev_1.predict(input_data)[0]
                st.metric(
                    label="Predicted Daily Revenue",
                    value=f"₹ {pred:,.2f}"
                )

        with tab2:
            if st.button("Predict Capped Revenue"):
                pred = rev_2.predict(input_data)[0]
                st.metric(
                    label="Capped Daily Revenue",
                    value=f"₹ {pred:,.2f}"
                )

        with tab3:
            if st.button("Predict Sales Category"):
                pred_class = clf.predict(input_data)[0]
                label = ["Low Sales Day", "Medium Sales Day", "High Sales Day"][pred_class]

                if pred_class == 2:
                    st.success(f"🔥 {label}")
                elif pred_class == 1:
                    st.warning(f"⚖️ {label}")
                else:
                    st.error(f"📉 {label}")

# ---------------- FOOTER ----------------
st.divider()


