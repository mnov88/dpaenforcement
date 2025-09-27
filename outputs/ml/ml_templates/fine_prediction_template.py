
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load data
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

# Separate features and targets
feature_cols = [col for col in train_df.columns if not any(x in col for x in ['fine_', 'severe_', 'remedy_', 'article5_', 'rights_', 'enforcement_'])]
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Two-stage fine prediction
# Stage 1: Predict if fine will be imposed
y_fine_imposed_train = train_df['fine_imposed']
y_fine_imposed_test = test_df['fine_imposed']

fine_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
fine_classifier.fit(X_train, y_fine_imposed_train)

# Evaluate binary classification
y_pred_imposed = fine_classifier.predict(X_test)
print("Fine Imposition Classification Report:")
print(classification_report(y_fine_imposed_test, y_pred_imposed))

# Stage 2: For cases where fine is imposed, predict amount
fine_cases_train = train_df[train_df['fine_imposed'] == 1]
fine_cases_test = test_df[test_df['fine_imposed'] == 1]

if len(fine_cases_train) > 0 and len(fine_cases_test) > 0:
    X_fine_train = fine_cases_train[feature_cols]
    X_fine_test = fine_cases_test[feature_cols]
    y_fine_amount_train = fine_cases_train['fine_amount_log']
    y_fine_amount_test = fine_cases_test['fine_amount_log']

    fine_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    fine_regressor.fit(X_fine_train, y_fine_amount_train)

    y_pred_amount = fine_regressor.predict(X_fine_test)

    print(f"\nFine Amount Prediction (R²): {r2_score(y_fine_amount_test, y_pred_amount):.3f}")
    print(f"Fine Amount Prediction (RMSE): {np.sqrt(mean_squared_error(y_fine_amount_test, y_pred_amount)):.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance_classification': fine_classifier.feature_importances_,
        'importance_regression': fine_regressor.feature_importances_
    }).sort_values('importance_classification', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

# Legal interpretation notes
print("\n" + "="*50)
print("LEGAL INTERPRETATION NOTES:")
print("="*50)
print("- This model predicts enforcement patterns, not legal correctness")
print("- Results should be validated against legal expertise")
print("- Consider country-specific legal traditions in interpretation")
print("- Temporal effects: early GDPR decisions may differ from current patterns")
