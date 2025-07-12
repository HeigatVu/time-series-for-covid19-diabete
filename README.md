# Medical Data Science: Time-Series Forecasting & Classification

This repository contains two data science projects focused on medical data, demonstrating key machine learning techniques for forecasting and classification.

1.  **COVID-19 Death Forecasting**: A time-series analysis to predict the number of future deaths in the US based on historical data.
2.  **Diabetes Classification**: A classic machine learning classification task to predict whether a patient has diabetes based on diagnostic measurements.

---

## 📈 Project 1: COVID-19 Death Forecasting

### 🎯 Objective
The goal of this project is to build a time-series forecasting model to predict the cumulative number of deaths from COVID-19 in the United States.

### 📊 Dataset
The analysis uses the `covid_19_clean_complete.csv` dataset, focusing on records for the "US" region. The primary feature used for forecasting is the daily cumulative number of deaths.

* **Time Period**: January 22, 2020 – July 27, 2020
* **Key Columns**: `Date`, `Country/Region`, `Deaths`

### ⚙️ Methodology

#### 1. Data Preprocessing
The core of this time-series problem is to transform the data into a format suitable for a supervised learning model.
* **Sliding Window**: A sliding window technique was applied to the time-series data. We use a sequence of 7 consecutive days' death counts (`seq_length = 7`) to predict the death count on the 8th day.
* **Train-Test Split**: The sequenced data was split into training and testing sets chronologically to ensure the model is tested on "future" data it has not seen.

#### 2. Modeling
A `LinearRegression` model from `scikit-learn` was used for this forecasting task. The model learns to predict the next day's value based on the linear relationship with the previous 7 days' values.

### 📈 Results
The model demonstrated high accuracy in predicting the trend of cumulative deaths in U.S, showing its effectiveness for this short-term forecasting task.

* **Test R² Score**: 0.998
* **Test Mean Absolute Error (MAE)**: ~322.5
* **Visualization**: The plot of actual vs. predicted values shows that the model's predictions closely follow the actual data trend.

![COVID-19 Death Forecast in U.S](https://github.com/HeigatVu/time-series-for-covid19-diabete/blob/main/covid19/death-prediction-us.png)

The other model demonstrated high accuracy in predicting the trend of cumulative recovered people around the world, showing its effectiveness for this short-term forecasting task.

* **Test R² Score**: 0.999
* **Test Mean Absolute Error (MAE)**: ~41117.29
* **Visualization**: The plot of actual vs. predicted values shows that the model's predictions closely follow the actual data trend.

![COVID-19 Recovered Forecast in the world](https://github.com/HeigatVu/time-series-for-covid19-diabete/blob/main/covid19/recovered-prediction-world.png)

---

## 🩺 Project 2: Diabetes Classification

### 🎯 Objective
To develop a machine learning model that can accurately classify whether a patient has diabetes based on several key diagnostic features.

### 📊 Dataset
This project uses the `diabetes.csv` dataset, which contains anonymized patient data.

* **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
* **Target**: `Outcome` (1 = Has Diabetes, 0 = Does Not Have Diabetes)

### ⚙️ Methodology

#### 1. Data Preprocessing
Data quality is critical in medical applications. The following steps were taken:
* **Handling Missing Values**: Certain columns like `Glucose`, `BloodPressure`, and `BMI` had zero values, which are physiologically impossible. These were identified and replaced with `NaN` to be treated as missing data.
* **Imputation**: The `NaN` values were then imputed using the **median** of their respective columns. The median is a robust choice as it is less sensitive to outliers than the mean.

#### 2. Modeling
A `RandomForestClassifier` from `scikit-learn` was chosen for this task. Random Forest is a powerful ensemble model that is well-suited for tabular data and provides insights into feature importance.

### 📈 Results
The model achieved a solid performance, demonstrating its potential as a diagnostic support tool.

* **Training Accuracy**: 100%
* **Testing Accuracy**: 75.3%

The high training accuracy and lower testing accuracy suggest some overfitting, which could be addressed with hyperparameter tuning (e.g., adjusting `max_depth` or `n_estimators`).

#### Feature Importance
The model identified `Glucose`, `BMI`, and `Age` as the most influential factors in predicting diabetes, which aligns with established clinical knowledge.

![Diabetes Feature Importance](https://github.com/HeigatVu/time-series-for-covid19-diabete/blob/main/diabet/feature-importances.png)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
