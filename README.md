# 🌫️ PM10 Prediction (Machine Learning + FastAPI)

This project focuses on predicting PM10 air pollution levels using time series data and machine learning.

The solution includes data processing, model training and a FastAPI backend for serving predictions.

---

## 📊 Dataset

The dataset contains real-world air quality measurements used for time series forecasting of PM10 values.

---

## 🧠 Machine Learning

The project includes:

- data preprocessing (handling missing values, normalization, feature engineering)  
- creation of time series input data  
- training and evaluation of prediction models  
- evaluation using regression metrics (MAE, MSE, MAPE, EVS)  

All experiments and model development are documented in the Jupyter Notebook.

---

## ⚙️ Backend (API)

A FastAPI backend is implemented to serve predictions.

### Endpoint

POST /predict

Input: JSON data  
Output:

{
  "prediction": 3.14
}

---

## 🐳 Technologies

- Python  
- FastAPI  
- Jupyter Notebook  
- Machine Learning (time series forecasting)  
- Docker  

---

## 📁 Project Structure

- api/ – FastAPI backend  
- project_pm10.ipynb – model development and experiments  
- dataset_E408.csv – dataset  

---

## 📚 Notes

This project was developed as part of a university assignment focused on time series prediction and backend development.
