# capstone_project-amex-credit-default-
# Predicting Credit Card Default likelihood: American Express
[![](https://img.shields.io/badge/Python-3.8-blue)](#) 

## Table of Contents
1. [Introduction](#introduction)
    - [Business Problem](#business-problem)
2. [Data Wrangling](#data-wrangling)
3. [Exploratory Data Analysis](#exploratory-data-analysis)
4. [Data Preprocessing and Training](#data-preprocessing-and-training)
    - [Multicollinearity: Feature Selection](#multicollinearity-feature-selection)
    - [Scaling Dataset: MinMaxScaler](#scaling-dataset-minmaxscaler)
    - [Class Imbalance: SMOTE](#class-imbalance-smote)
5. [Modeling](#modeling)
    - [Metrics](#metrics)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Model Selection](#model-selection)
6. [Conclusions and Recommendations](#conclusions-and-recommendations)
7. [References](#references)
8. [Installation](#installation)
9. [Technologies Used](#technologies-used)
10. [Contact](#contact)

## Introduction
This project seeks to address the significant challenge of credit default faced by American Express by leveraging data science. By developing a predictive model, we aim to forecast the likelihood of a customer defaulting on their credit card payments, thereby aiding American Express in managing credit default risks more effectively.

### Business Problem
Credit defaults pose substantial financial risks to American Express. The goal is to harness data science to create a classification model using 2021 customer data to predict credit card payment defaults, enabling proactive mitigation strategies and fostering a financially secure customer-issuer relationship.

## Data Wrangling
We performed data cleaning and preparation by loading data from a CSV file, standardizing nomenclatures, handling missing values, and profiling the dataset to ensure optimal design for future analysis steps.
- [Data Wrangling Notebook](/notebooks/data_wrangling_2capstone.ipynb)

## Exploratory Data Analysis
We visualized the dataset to understand the inherent dynamics within the data, identify multicollinearity, and observe class imbalance issues which are crucial for the next steps of preprocessing and modeling.
- [EDA Notebook](/notebooks/eda_amex_project.ipynb)

## Data Preprocessing and Training
In this stage, multicollinearity was addressed by dropping redundant features, the dataset was scaled using MinMaxScaler, and the class imbalance was handled using SMOTE to ensure a balanced dataset for effective modeling.

### Multicollinearity: Feature Selection
Addressed multicollinearity by dropping features exhibiting high correlation to reduce redundancy and improve model performance.

### Scaling Dataset: MinMaxScaler
Employed MinMaxScaler to harmonize the range of features, ensuring each feature has an equal opportunity to influence the model.

### Class Imbalance: SMOTE
We utilized SMOTE to handle class imbalance, enhancing the dataset with more instances of the minority class for a balanced training set.
- [Preprocessing and Training Notebook](notebooks/preprocessing_amex.ipynb)

## Modeling
The modeling phase involved evaluating different metrics, hyperparameter tuning, and selecting the XGBoost Classifier due to its high performance, efficiency, and suitability for handling imbalanced datasets.

### Metrics
Employed AUC-ROC and AUC-PRC as primary metrics to evaluate model performance, with AUC-ROC used as a baseline for model selection due to its robustness across various thresholds.

### Hyperparameter Tuning
Used RandomizedSearchCV for efficient hyperparameter tuning, optimizing the model's learning characteristics without exhaustive computational demand.

### Model Selection
In this section, we evaluated the models based on their performance in an unseen test set and selected the best model to apply. 
- [Modeling Notebook](notebooks/modeling_amex.ipynb)

## Conclusions and Recommendations
The analysis underscores the critical challenge of credit defaults and highlights the efficacy of the selected model in addressing this issue. Recommendations include continuous model evaluation, further feature engineering, devising risk mitigation strategies, and ensuring model interpretability, fairness, and regulatory compliance.

## References
The references section lists all the external resources and data sources referred to in the project

## Installation
The project was implemented in Python 3.8. To install the required packages, use the following command:
```bash
pip install -r requirements.txt
```
## Technologies Used:
- Python
- Numpy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

## Contact
If you have any questions, comments, or would like to contribute, please feel free to contact me at camilodurangos@gmail.com.
