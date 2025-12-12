#change the input data 
#save and load encoder , scaler and smote
#save and load classififer and put correct name in the button section in end. 
##python3 -m streamlit run app.py

import streamlit as st
import pandas as pd
import pickle
import numpy as np

#load models
with open("model_1.pkl", "rb") as f:
    rev_1 = pickle.load(f)
with open("model_2_capped.pkl", "rb") as f:
    rev_2 = pickle.load(f)
with open("sales_classifier.pkl","rb") as f:
    clf = pickle.load(f)




st.set_page_config(page_title="Smart Retail Insight Engine", layout="centered")
st.title("🛒 Smart Retail Insight Engine")
st.subheader("Random Forest Models for Revenue and Sales Day Classification")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info("""
This app predicts:
1. Item revenue
2. Sales day status

Models used:
- Random Forest Regressor
- Random Forest Classifier
""")

st.header("Enter Date Details for Daily Revenue Prediction")

# User inputs
day = st.number_input("Day (1–31)", min_value=1, max_value=31)
month = st.number_input("Month (1–12)", min_value=1, max_value=12)
year = st.number_input("Year (e.g., 2017)", min_value=2000, max_value=2030)

# Derived features
import datetime
try:
    date_obj = datetime.date(int(year), int(month), int(day))
    dayofweek = date_obj.weekday()       # 0 = Monday
    is_weekend = 1 if dayofweek >= 5 else 0
except:
    st.error("Invalid date!")

# LAG INPUTS
revenue_lag_1 = st.number_input("Revenue 1 Day Before", min_value=0.0, step=100.0)
revenue_lag_7 = st.number_input("Revenue 7 Days Before", min_value=0.0, step=100.0)

# Prepare dataframe
input_data = pd.DataFrame({
    "day": [day],
    "month": [month],
    "year": [year],
    "dayofweek": [dayofweek],
    "is_weekend": [is_weekend],
    "revenue_lag_1": [revenue_lag_1],
    "revenue_lag_7": [revenue_lag_7]
})

# Encode categorical
#input_data["product_category_name"] = encoders["product_category_name"].transform(input_data["product_category_name"])

# Scale input
#input_scaled = scaler.transform(input_data)

# Predictions
if st.button("Predict Daily Revenue (Model 1)"):
    pred = rev_1.predict(input_data)[0]
    st.success(f"💰 Predicted Revenue: {pred:.2f}")

if st.button("Predict Daily Revenue (Model 2 - Capped)"):
    pred = rev_2.predict(input_data)[0]
    st.success(f"💰 Predicted Revenue (Capped): {pred:.2f}")

if st.button("Predict Sales Category"):
    pred_class = clf.predict(input_data)[0]
    
    # Convert numeric class to text
    if pred_class == 0:
        label = "Low Sales Day"
    elif pred_class == 1:
        label = "Medium Sales Day"
    else:
        label = "High Sales Day"
    
    st.success(f"📈 Predicted Sales Category: {label}")

