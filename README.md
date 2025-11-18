# Hotel Haven – Predicting Booking Cancellations with Early Risk Signals to Protect Revenue

![Overview](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure1.png)

---

## Overview

Hotel Haven, a luxury hotel chain, faces a critical operational challenge: a **32.77% cancellation rate** — nearly **1 in every 3 bookings**.  
This unpredictability leads to lost revenue, underutilized resources, and missed opportunities to retain guests.

This project builds a machine learning pipeline that predicts which bookings are likely to cancel.  
It enables the hotel to **intervene early**, prevent losses, and make **data-backed staffing and pricing decisions**.

---

## Problem Statement

Hotel Haven’s existing systems do not offer foresight into why or when guests cancel.  
The business needed a model to:

- Predict cancellations at booking time  
- Quantify risk with calibrated probabilities  
- Drive targeted retention strategies  
- Improve operational efficiency and customer experience

![Problem Statement](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure2.png)

---

## Business Impact

**Cancellations Analyzed**:

- **Total bookings:** **36,285**  
- **Cancelled stays:** **11,889**  
- **Revenue at risk:** **$1.23M**  
*(based on an average daily rate of $103.42)*

![Business Impact](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure3.png)

---

### If the model recovers even 10–15% of high-risk cancellations:

- **1,200–1,800 stays** retained  
- **$124K–$186K** in monthly revenue protected  
- **Occupancy ↑ by 0.5–1.0 percentage points**  
- **Staffing, perks, and pricing resources** allocated with greater precision

---

## Project Objectives

1. Build a predictive model to flag cancellation risk at booking  
2. Identify key behavioral and financial drivers of cancellations  
3. Calibrate thresholds for intervention  
4. Segment guests into **Low**, **Medium**, and **High** risk bands  
5. Create a playbook for cost-efficient retention strategies

---

## Models Evaluated

| Model               | ROC–AUC | PR–AUC | F1 Score |
|---------------------|---------|--------|----------|
| Logistic Regression | ~**0.88** | ~**0.82** | ~**0.76** |
| Random Forest       | ~**0.90** | ~**0.84** | ~**0.77** |
| XGBoost             | ~**0.92** | ~**0.87** | ~**0.79** |
| **LightGBM (final)**| **0.936** | **0.896** | **0.801** |

**Chosen Model:** LightGBM was selected for its **high performance**, **fast training**, and **effective handling of categorical data** and **class imbalance**.

![Model Comparison](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure5.png)  
![LightGBM Output](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure6.png)

---

## Key Features

Top predictive features based on permutation importance:

- **Lead Time** (days between booking and check-in)  
- **Special Requests Count**  
- **Average Daily Rate (ADR)**  
- **Market Segment Type** (e.g., OTA vs Direct)  
- **Room Type**, **Parking Space**, **Customer Type**

![Key Features](https://github.com/DesireeSalvant/DetectingHotelBookingCancelationsMachineLearning/blob/main/hotelfigure4.png)

---

## Risk Band Strategy

| Risk Tier    | Score Range | Action Strategy                             |
|--------------|-------------|---------------------------------------------|
| **High Risk**   | ≥ **0.70**     | Call/SMS outreach, perks, reassurance       |
| **Medium Risk** | **0.36–0.69**  | Reminder emails, itinerary nudges          |
| **Low Risk**    | < **0.36**     | Standard confirmations only                |

Risk thresholds are calibrated using model predictions on the test set.

---

## Technical Stack

**Environment**  
- Google Colab (Python 3.10)

**Core Libraries**  
- `pandas`, `numpy` – Data cleaning and manipulation  
- `scikit-learn` – Model evaluation and baselines  
- `lightgbm`, `xgboost`, `randomforest` – Tree-based models  
- `matplotlib`, `seaborn` – Visualizations  
- `shap`, `permutation_importance` – Model explainability  

**Pipeline Components**  
- One-hot encoding for categorical variables  
- Balanced class weights for Logistic Regression  
- F1-optimal thresholding for classification  
- AUC, PR-AUC, Precision, Recall, Confusion Matrix

---

## Next Steps

- Deploy via Flask or FastAPI as a RESTful scoring API  
- Integrate into CRM or hotel booking system backend  
- Build a Tableau or Streamlit dashboard to track model and business KPIs  
- Automate quarterly retraining with new bookings data  
- AB test retention strategies by risk segment

---

## Author

**Desiree Salvant**  
Executive Data Scientist · *The Coding Executive™*  





