# Sales Category Prediction

A machine learning project that classifies e-commerce sales transactions into product **categories** using order-level features (amount, profit, quantity, payment mode, location, etc.). Multiple classification algorithms are trained and compared.

## Project Overview

- **Task:** Multi-class classification — predict `Category` (e.g. Electronics, Clothing, Furniture...) from transaction data
- **Dataset:** Sales Dataset (Kaggle) — order-level e-commerce sales records
- **Approach:** EDA → outlier inspection → categorical encoding → train/test split → model comparison

## Workflow

1. **Data loading & cleaning** — inspect shape, duplicates, nulls, dtypes; parse `Order Date`
2. **EDA** — outlier checks (box plots for Amount, Profit, Quantity) and category distribution visualizations
3. **Preprocessing** — drop identifier columns (`Order ID`, `Order Date`, `CustomerName`, `Year-Month`), drop nulls, label-encode categorical columns (`Sub-Category`, `PaymentMode`, `State`, `City`)
4. **Modeling** — train/test split (80/20), then train and evaluate five classifiers with accuracy score and confusion matrix for each

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.3891 |
| K-Nearest Neighbors | 0.4854 |
| Random Forest | 0.9958 |
| Decision Tree | 1.0000 |
| Naive Bayes | 0.4854 |
| SVM | 0.3305 |

Tree-based models (Decision Tree, Random Forest) substantially outperform the linear/distance-based models on this dataset, since `Category` is closely tied to `Sub-Category` and other encoded features they can split on directly.

> **Note:** Decision Tree hitting 1.0000 accuracy is a strong signal of data leakage (e.g. `Sub-Category` almost perfectly determines `Category`) rather than a production-ready result. Worth revisiting with `Sub-Category` excluded from the features to get a more realistic estimate.

## Tech Stack

- Python 3
- pandas, numpy — data handling
- matplotlib, seaborn — visualization
- scikit-learn — LogisticRegression, KNeighborsClassifier, RandomForestClassifier, DecisionTreeClassifier, GaussianNB, SVC

## Repository Structure

```
sales-prediction/
├── sales-prediction.ipynb   # full EDA + modeling notebook
├── images
└── README.md
```


## Author

Sohila — Data Science graduate, Sadat Academy for Management Sciences
