# Smart_Retail_Insight-Engine
Predicting Daily Revenue & Classifying High vs Low Sales Days Using Machine Learning

## Project Overview

This project analyzes and models e-commerce sales data to predict daily revenue and classify sales performance using machine learning.
The goal is to help businesses understand future revenue trends and identify low, medium, and high sales days for better planning.

## Objectives

Predict daily total revenue using historical sales data

Classify days into Low / Medium / High sales categories

Handle noisy real-world data using outlier capping

Deploy models using a Streamlit web application

## Approach & Methodology
### 1. Data Aggregation

Transaction-level data is aggregated to daily level

Daily metrics include:

Total revenue

Lagged revenues (1 day, 7 days)

Calendar features (day, month, weekday, weekend)

### 2. Feature Engineering

To capture temporal patterns, we created:

Lag features

Revenue from previous day (lag_1)

Revenue from previous week (lag_7)

Calendar features

Day of week

Weekend indicator

Month and year

This allows the model to learn both short-term trends and seasonal behavior.

### 3. Outlier Handling (Capping)

Daily revenue contains extreme values caused by:

Festivals

Flash sales

Promotional events

Instead of removing these values, we applied capping:

Extreme values are limited to percentile boundaries

This reduces noise while preserving all data points

## Result:
R² improved from 0.44 → 0.56 after capping.

### 4. Revenue Prediction (Regression)

Model used: Random Forest Regressor

Input: lag features + calendar features

Output: predicted daily total revenue

This model captures non-linear patterns in real-world sales data.

### 5. Sales Classification

Days are classified into:

Low sales

Medium sales

High sales

Classification is based on:

Temporal features

Historical sales patterns

This provides a business-friendly interpretation of sales performance.

### 6. Model Deployment

Models are saved using pickle

A Streamlit application allows:

User input of features

Revenue prediction

Sales category prediction

## Evaluation Metrics
Regression

MAE

RMSE

R² Score

Classification

Accuracy

Precision, Recall, F1-Score

Confusion Matrix

## Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Streamlit

## Future Work

Implement dedicated time-series models (ARIMA / LSTM)

Add rolling statistics for smoother trend learning

Improve classification using revenue-based thresholds

Deploy the app online

## Conclusion

This project demonstrates how machine learning with temporal features can be used to predict and interpret sales behavior in e-commerce.
By combining regression, classification, and visualization, it provides actionable insights for business decision-making.
