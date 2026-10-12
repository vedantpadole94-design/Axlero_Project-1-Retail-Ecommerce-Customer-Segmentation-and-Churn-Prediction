# Axlero Project 1 — Retail & Ecommerce Customer Segmentation & Churn Prediction

## Problem Statement
Axlero's D2C retail partners face high acquisition costs and unpredictable retention.
This project segments customers using RFM analysis and predicts 30-day churn.

## Success Metrics
- Data retention >= 95% after cleaning
- Churn model ROC-AUC >= 0.75
- At least 4 actionable customer segments

## Structure
- src/data/       ingestion and SQL extraction
- src/features.py RFM engineering
- src/model.py    churn classifiers
- notebooks/      EDA, clustering, modeling
- reports/        figures and final report

## Setup
    pip install -r requirements.txt
    createdb axlero_retail
    python src/data/load_db.py
    make features
    make train

## Author
Vedant Padole
