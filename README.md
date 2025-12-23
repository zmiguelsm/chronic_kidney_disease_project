# Chronic Kidney Disease Detection:
This repo contains an end-to-end machine learning pipeline for diagnosing **Chronic Kidney Disease (CKD)** using clinical and demographic data.

## Project Overview
The goal of this project is to classify patients as CKD or non-CKD based on multiple health indicators. The analysis covers:

- Data cleaning and preprocessing
- Exploratory data analysis
- Handling class imbalance
- Model training, evaluation, and interpretation

## Models Implemented
- **Logistic Regression**: interpretable linear model for baseline performance
- **Random Forest**: non-linear model capturing complex feature interactions
- **XGBoost (with SMOTE)**: gradient boosting model optimized for imbalanced data


## Results (Test Set Highlights)
- **Logistic Regression**: moderate performance, strong interpretability
- **Random Forest**: improved non-linear capture, but some bias toward majority class. Tuned `mtry` using OOB error and interpreted feature importance with Gini and accuracy decrease.
- **XGBoost + SMOTE**: best overall performance with highest ROC–AUC

## Tools & Packages
- **Language**: R
- **Packages**: tidyverse, caret, randomForest, xgboost, smotefamily, pROC, ggplot2, recipes, themis...

