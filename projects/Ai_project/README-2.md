# AI-Project

# Beat Classifier for PPG Signals

##Table of Contents

Introduction
Team Members
Project Structure
Setup and Installation
Data Overview
Preprocessing
Model Training and Evaluation
Results
Contributions
Acknowledgments

## Introduction

Photoplethysmography (PPG) signal classification is essential in biomedical engineering for diagnosing and monitoring heart conditions. This project aims to develop a beat classifier for PPG signals, classifying each beat into Normal (N), Supraventricular (S), and Ventricular (V). Accurate classification is vital for detecting arrhythmias and other cardiac anomalies.

## Team Members

Drmic A.
Javadi M.
Shala M.

## Project Structure

heart_beat_classification.ipynb: Notebook containing the preprocessing, model training, and evaluation.
Drmic_Javadi_Shala.pdf: Detailed report describing the methodology and results.
Setup and Installation

### Clone the repository:

```
git clone <repository_url>
cd <repository_directory>
```

### Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
### Install the required packages:
```
pip install -r requirements.txt
```
## Data Overview

The dataset consists of PPG recordings from 105 patients with different sampling frequencies (128Hz and 250Hz). The goal is to classify heartbeats into three categories: Normal (N), Supraventricular (S), and Ventricular (V). The dataset is imbalanced, with the majority of beats being normal.

## Preprocessing

Data Splitting and Stratified Sampling
The dataset is divided into training, validation, and testing sets using stratified sampling to maintain class distribution.
The training set is used to train the model, the validation set to monitor performance during training, and the testing set to evaluate the model's generalization capability.
Signal Resampling
Signals recorded at 128 Hz are resampled to 250 Hz for uniformity.
Normalization
Both training and validation sets are normalized to improve the convergence speed of learning algorithms.
Noise Cancellation
A Butterworth bandpass filter (0.25 Hz to 10 Hz) is applied to reduce noise.
High amplitude oscillations are manually inspected and excluded if necessary.
Peak Segmentation
Peaks are segmented by creating windows centered around each peak, extending 250 samples on either side.
Valid Segments
Segments with excessively high peak-to-peak amplitudes are discarded to ensure a cleaner and more reliable dataset.
Model Training and Evaluation

## Models
VGG-Style CNN
LSTM Model
Bidirectional LSTM Model
1D CNN Model

## Metrics
Precision
Recall
Specificity
Adaptive Metric Adjustment Strategy
Weighted Accuracy

## Results

The VGG-Style CNN achieved the highest weighted accuracy (92.32%) and recall (0.8350) for multiclass classification.
The LSTM model achieved the highest weighted accuracy (70.62%) and recall (0.9052) for binary classification.

## Contributions

Shala M.: Data preprocessing, model training
Javadi M.: Model evaluation, report preparation
Drmic A.: Project coordination, methodology development

## Acknowledgments

We acknowledge the guidance and support provided by our course instructors and peers in the biomedical engineering department.
