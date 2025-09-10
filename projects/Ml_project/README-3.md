# ML_project

# Flood Prediction using Machine Learning

## Table of Contents

Introduction
Team Members
Project Structure
Setup and Installation
Data Overview
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Results
Contributions
Acknowledgments

## Introduction

This project involves developing a machine learning model to predict the outcome of flood events based on initial conditions. The model is trained on 3,000 simulated flood incidents in an urban area of Houston, TX, with the goal of replicating the predictions of the simulator that generated these incidents.

## Team Members

Mohamed Shala

## Project Structure

ML.ipynb: Notebook containing the machine learning analysis, model training, and evaluation.
Assignment Description.pdf: Detailed description of the assignment and dataset.
training_parameters.csv: CSV file containing parameters influencing flood simulations for the training set.
test_parameters.csv: CSV file containing parameters for the test set.
edge_info.csv: CSV file with geographical data of the street segments.
training/: Directory containing CSV files with initial and final states of each training observation.
test/: Directory containing CSV files with initial states for the test set.
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
## Data Overview

The dataset consists of:

Nodes: Intersections or endpoints of streets.
Edges: Street segments linking two nodes, defined by 'head_id' and 'tail_id'.
training_parameters.csv and test_parameters.csv: Files containing simulation parameters for each observation.
edge_info.csv: File containing longitude, latitude, and altitude of each edge's center.
training/ and test/ directories: Containing individual CSV files for each observation, recording various edges with attributes like initial and final flood states.

## Exploratory Data Analysis (EDA)

EDA was conducted to understand the distribution and characteristics of the dataset. Visualization techniques such as line plots and scatter plots were used to analyze trends, correlations, and patterns.

## Model Training and Evaluation

Various machine learning models were trained and evaluated, including:

Logistic Regression
Random Forest
Gradient Boosting
Neural Networks
The models were evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Results

The performance of the models is documented in the ML.ipynb notebook. The best performing model's predictions for the test set are saved in individual CSV files, mirroring the structure of the training set.


## Acknowledgments

We acknowledge the guidance and support provided by our course instructor, Dr. Masoud Jalayer, for the machine learning assignment at [Your Institution Name].

