# Online Shoppers Purchase Prediction (Decision Tree)

This project uses a Decision Tree classifier to predict whether an online shopper will make a purchase (`Revenue`) based on their session behaviour on an e-commerce website.

## Dataset

- Rows: 12,330 sessions
- Target: `Revenue` (0 = No Purchase, 1 = Purchase)
- Feature groups:
  - Administrative / Administrative_Duration
  - Informational / Informational_Duration
  - ProductRelated / ProductRelated_Duration
  - BounceRates, ExitRates, PageValues, SpecialDay
  - Month, OperatingSystems, Browser, Region, TrafficType
  - VisitorType, Weekend

## Method

1. Train–test split with stratification on `Revenue`.
2. Preprocessing using `ColumnTransformer`:
   - Numeric features → optionally scaled
   - Categorical features (`Month`, `VisitorType`, etc.) → OneHotEncoder
3. Model: `DecisionTreeClassifier` with pruning
   - Tuned `max_depth` and `min_samples_leaf` using `GridSearchCV`.
   - `class_weight="balanced"` to handle class imbalance.
4. Evaluation:
   - Main metric: F1 score for the positive class (`Revenue = 1`).
   - Classification report and confusion matrix.

## Results

- Best cross-validated F1 (GridSearchCV): ~0.66
- Test F1 score for purchase class: ~0.64

## How to run

```bash
pip install -r requirements.txt
jupyter lab