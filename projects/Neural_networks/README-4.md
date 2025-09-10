# Neural_Networks
# Time Series Forecasting

## Table of Contents

Introduction
Team Members
Project Structure
Setup and Installation
Data Preparation
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Results
Contributions
Acknowledgments

## Introduction

This project involves the application of time series forecasting techniques to predict future values based on historical data. The work includes data preprocessing, exploratory data analysis, and training various forecasting models to achieve accurate predictions.

## Team Members

Mohammad Amiri (10887256)
Sara Limooee (100886949)
Dorsa Moadeli (10926114)
Mohamed Shoala (10871548)

## Project Structure

Time_Series_Forecasting.ipynb: Notebook containing the time series forecasting analysis and model training.
Setup and Installation

Clone the repository:
```
git clone <repository_url>
cd <repository_directory>
```
Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install the required packages:
```
pip install -r requirements.txt
```
## Data Preparation

The dataset consists of time series data that has been preprocessed to handle missing values, normalize the data, and create necessary time-based features. Steps include:

Handling missing values (imputation)
Normalization/Standardization
Feature engineering (e.g., creating lag features, rolling statistics)
Exploratory Data Analysis (EDA)

EDA was conducted to understand the time series data's trends, seasonality, and autocorrelation patterns. Visualization techniques such as line plots, autocorrelation plots, and seasonal decomposition were used. Details are available in Time_Series_Forecasting.ipynb.

## Model Training and Evaluation

Various time series forecasting models were trained and evaluated, including:

ARIMA (AutoRegressive Integrated Moving Average)
SARIMA (Seasonal ARIMA)
Exponential Smoothing (Holt-Winters)
Prophet
LSTM (Long Short-Term Memory) networks

## Results

The performance of the models was evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). Detailed results and visualizations are documented in the Time_Series_Forecasting.ipynb notebook.

## Contributions

Mohammad Amiri (10887256): Data preprocessing and cleaning
Sara Limooee (100886949): Exploratory data analysis and visualization
Mohamed Shoala (10871548): Model training and evaluation
Dorsa Moadeli (10926114): Documentation and report preparation

## Acknowledgments

We acknowledge the guidance and support provided by our course instructors and peers.

This README provides a comprehensive overview of your Time Series Forecasting project, guiding users through its structure and content. Be sure to replace placeholders with actual team member names and any additional relevant details.
