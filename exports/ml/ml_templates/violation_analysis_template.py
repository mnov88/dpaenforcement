
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv('train.csv', index_col=0)
test_df = pd.read_csv('test.csv', index_col=0)

# Features for violation prediction (exclude enforcement outcomes)
feature_cols = [col for col in train_df.columns if not any(x in col for x in ['fine_', 'severe_', 'remedy_', 'enforcement_outcome'])]
X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Multi-label violation prediction
violation_targets = ['article5_violation', 'rights_violation']
y_train = train_df[violation_targets]
y_test = test_df[violation_targets]

# Multi-output classifier
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Evaluation
for i, target in enumerate(violation_targets):
    print(f"\n{target.upper()} CLASSIFICATION REPORT:")
    print("="*50)
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_article5': classifier.estimators_[0].feature_importances_,
    'importance_rights': classifier.estimators_[1].feature_importances_
}).sort_values('importance_article5', ascending=False)

print("\nTOP 15 FEATURES FOR VIOLATION PREDICTION:")
print(feature_importance.head(15))

# Correlation analysis between violation types
violation_corr = y_test.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(violation_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Violation Types')
plt.tight_layout()
plt.savefig('violation_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("LEGAL ANALYSIS INSIGHTS:")
print("="*60)
print("- High correlation between violation types suggests systemic compliance issues")
print("- Feature importance indicates key risk factors for violations")
print("- Results can inform proactive compliance strategies")
