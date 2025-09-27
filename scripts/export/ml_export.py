from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .base import BaseExporter


class MLFeatureExporter(BaseExporter):
    """Export ML-ready features with embeddings and proper train/test splits."""

    def __init__(self, wide_csv: Path, long_tables_dir: Optional[Path] = None):
        super().__init__(wide_csv, long_tables_dir)
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def export(self, output_dir: Path,
               embeddings_model: str = 'all-MiniLM-L6-v2',
               test_size: float = 0.2,
               random_state: int = 42) -> None:
        """Export ML-ready features with embeddings and splits."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare feature matrix
        features_df = self._create_feature_matrix()

        # Create target variables
        targets_df = self._create_target_variables()

        # Generate text embeddings
        embeddings_df = self._create_text_embeddings(embeddings_model, output_dir)

        # Combine all features
        ml_dataset = self._combine_features(features_df, targets_df, embeddings_df)

        # Create stratified splits
        splits = self._create_stratified_splits(ml_dataset, test_size, random_state)

        # Export datasets
        self._export_datasets(splits, output_dir)

        # Export feature metadata
        self._export_feature_metadata(ml_dataset, output_dir)

        # Export preprocessing artifacts
        self._export_preprocessing_artifacts(output_dir)

        # Create ML pipeline templates
        self._create_ml_templates(output_dir)

    def _create_feature_matrix(self) -> pd.DataFrame:
        """Create ML-ready feature matrix."""
        df = self.df.copy()

        # Start with clean feature set
        features = pd.DataFrame({'decision_id': df['decision_id']})

        # Geographic features
        features['country_eu'] = (df['country_group'] == 'EU').astype(int)
        features['country_eea'] = (df['country_group'] == 'EEA').astype(int)
        features['country_non_eea'] = (df['country_group'] == 'NON_EEA').astype(int)

        # Temporal features
        if 'decision_year' in df.columns:
            features['decision_year'] = df['decision_year']
            features['years_since_gdpr'] = df['decision_year'] - 2018
            features['decision_year_normalized'] = (df['decision_year'] - df['decision_year'].min()) / (df['decision_year'].max() - df['decision_year'].min())

        if 'decision_quarter' in df.columns:
            features['quarter_q1'] = (df['decision_quarter'] == 'Q1').astype(int)
            features['quarter_q2'] = (df['decision_quarter'] == 'Q2').astype(int)
            features['quarter_q3'] = (df['decision_quarter'] == 'Q3').astype(int)
            features['quarter_q4'] = (df['decision_quarter'] == 'Q4').astype(int)

        # Case type features
        features['breach_case'] = df.get('breach_case', 0).astype(int)
        features['cross_border_case'] = self._create_cross_border_indicator(df)

        # Legal complexity features
        features['n_principles_discussed'] = df.get('n_principles_discussed', 0)
        features['n_principles_violated'] = df.get('n_principles_violated', 0)
        features['n_corrective_measures'] = df.get('n_corrective_measures', 0)

        # Violation type features (from multi-select columns)
        violation_features = self._extract_violation_features(df)
        features = features.join(violation_features, on='decision_id')

        # Rights violation features
        rights_features = self._extract_rights_features(df)
        features = features.join(rights_features, on='decision_id')

        # Enforcement features
        enforcement_features = self._extract_enforcement_features(df)
        features = features.join(enforcement_features, on='decision_id')

        # Economic features (if available)
        if 'turnover_eur' in df.columns:
            features['has_turnover_info'] = (~df['turnover_eur'].isna()).astype(int)
            features['turnover_log1p'] = np.log1p(df['turnover_eur'].fillna(0))

        # Derived complexity indices
        features['legal_complexity_index'] = (
            features['n_principles_discussed'] +
            features['n_principles_violated'] +
            features['n_corrective_measures']
        )

        features['violation_severity_index'] = self._create_severity_index(df)

        # DPA characteristics (encoded)
        if 'dpa_name_canonical' in df.columns:
            dpa_features = self._create_dpa_features(df)
            features = features.join(dpa_features, on='decision_id')

        # Sector features (if ISIC available)
        if 'isic_section' in df.columns:
            sector_features = self._create_sector_features(df)
            features = features.join(sector_features, on='decision_id')

        return features.set_index('decision_id')

    def _create_target_variables(self) -> pd.DataFrame:
        """Create various target variables for ML tasks."""
        df = self.df.copy()
        targets = pd.DataFrame({'decision_id': df['decision_id']})

        # Fine-related targets
        if 'fine_eur' in df.columns:
            targets['fine_imposed'] = (df['fine_eur'] > 0).astype(int)
            targets['fine_amount_log'] = np.log1p(df['fine_eur'])

            # Fine amount categories
            fine_conditions = [
                (df['fine_eur'] == 0),
                (df['fine_eur'] > 0) & (df['fine_eur'] <= 10000),
                (df['fine_eur'] > 10000) & (df['fine_eur'] <= 100000),
                (df['fine_eur'] > 100000) & (df['fine_eur'] <= 1000000),
                (df['fine_eur'] > 1000000)
            ]
            targets['fine_category'] = np.select(fine_conditions, [0, 1, 2, 3, 4], default=-1)

        # Severity targets
        targets['severe_measures'] = df.get('severity_measures_present', 0).astype(int)
        targets['remedy_only'] = df.get('remedy_only_case', 0).astype(int)

        # Violation type prediction targets
        targets['article5_violation'] = self._create_article5_target(df)
        targets['rights_violation'] = self._create_rights_target(df)

        # Multi-class enforcement outcome
        targets['enforcement_outcome'] = self._create_enforcement_outcome_target(df)

        return targets.set_index('decision_id')

    def _create_text_embeddings(self, model_name: str, output_dir: Path) -> pd.DataFrame:
        """Create text embeddings for narrative fields."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            warnings.warn("sentence-transformers not available, skipping embeddings")
            return pd.DataFrame({'decision_id': self.df['decision_id']}).set_index('decision_id')

        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            warnings.warn(f"Could not load embeddings model {model_name}: {e}")
            return pd.DataFrame({'decision_id': self.df['decision_id']}).set_index('decision_id')

        df = self.df
        embeddings_data = {'decision_id': df['decision_id']}

        # Text fields to embed
        text_fields = {
            'q36_text': 'Article 5/6 findings summary',
            'q52_text': 'Fine calculation reasoning',
            'q67_text': 'EDPB guidelines references',
            'q68_text': 'Expert case summary'
        }

        for field, description in text_fields.items():
            if field in df.columns:
                # Clean and prepare text
                texts = df[field].fillna('').astype(str)
                texts = texts.replace('', '[NO_TEXT]')  # Handle empty strings

                # Generate embeddings
                try:
                    embeddings = model.encode(texts.tolist(), show_progress_bar=True)

                    # Add embeddings as columns
                    for i in range(embeddings.shape[1]):
                        embeddings_data[f'{field}_emb_{i:03d}'] = embeddings[:, i]

                    print(f"Generated {embeddings.shape[1]} embeddings for {field}")

                except Exception as e:
                    warnings.warn(f"Could not generate embeddings for {field}: {e}")
                    continue

        # Save embeddings model info
        embeddings_info = {
            'model_name': model_name,
            'embedding_dimension': embeddings.shape[1] if 'embeddings' in locals() else 0,
            'text_fields_embedded': list(text_fields.keys()),
            'preprocessing_notes': [
                'Empty strings replaced with [NO_TEXT]',
                'All text normalized to string type',
                'Embeddings generated using sentence-transformers'
            ]
        }

        with open(output_dir / 'embeddings_info.json', 'w') as f:
            json.dump(embeddings_info, f, indent=2)

        return pd.DataFrame(embeddings_data).set_index('decision_id')

    def _combine_features(self, features_df: pd.DataFrame, targets_df: pd.DataFrame,
                         embeddings_df: pd.DataFrame) -> pd.DataFrame:
        """Combine all feature types into a single dataset."""
        # Start with features
        combined = features_df.copy()

        # Add targets
        combined = combined.join(targets_df, how='left')

        # Add embeddings
        combined = combined.join(embeddings_df, how='left')

        # Fill NaN values appropriately - preserve targets to avoid leakage
        numeric_cols = combined.select_dtypes(include=[np.number]).columns

        # Identify target columns (fine/turnover amounts, ratios, logs)
        target_cols = [col for col in numeric_cols if any(x in col for x in
                      ['fine_eur', 'turnover_eur', '_ratio', '_log1p', 'fine_amount', 'turnover_amount'])]

        for col in numeric_cols:
            if '_emb_' in col:  # Embedding columns
                combined[col] = combined[col].fillna(0)
            elif col in target_cols:  # Target variables - preserve nulls, add missing indicators
                combined[f'{col}_missing'] = combined[col].isnull().astype(int)
                # Don't impute targets - preserve legal semantics of "not assessed"
            else:  # Feature columns - safe to impute
                combined[col] = combined[col].fillna(combined[col].median())

        return combined

    def _create_stratified_splits(self, dataset: pd.DataFrame, test_size: float,
                                random_state: int) -> Dict[str, pd.DataFrame]:
        """Create stratified train/test splits preserving legal distributions."""
        # Create stratification variable combining key characteristics
        stratify_vars = []

        if 'fine_imposed' in dataset.columns:
            stratify_vars.append(dataset['fine_imposed'].astype(str))

        if 'country_eu' in dataset.columns:
            country_group = (
                dataset['country_eu'].astype(str) + '_' +
                dataset['country_eea'].astype(str) + '_' +
                dataset['country_non_eea'].astype(str)
            )
            stratify_vars.append(country_group)

        if 'breach_case' in dataset.columns:
            stratify_vars.append(dataset['breach_case'].astype(str))

        # Combine stratification variables
        if stratify_vars:
            stratify_key = stratify_vars[0]
            for var in stratify_vars[1:]:
                stratify_key = stratify_key + '_' + var
        else:
            stratify_key = None

        # Separate features and targets
        feature_cols = [col for col in dataset.columns if not self._is_target_column(col)]
        target_cols = [col for col in dataset.columns if self._is_target_column(col)]

        X = dataset[feature_cols]
        y = dataset[target_cols] if target_cols else None

        # Create splits
        if stratify_key is not None:
            # Only stratify if we have enough samples per group
            value_counts = stratify_key.value_counts()
            if value_counts.min() >= 2:  # Need at least 2 samples per group
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state,
                        stratify=stratify_key
                    )
                except ValueError:
                    # Fallback to random split if stratification fails
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Combine features and targets back
        train_data = X_train.join(y_train) if y_train is not None else X_train
        test_data = X_test.join(y_test) if y_test is not None else X_test

        # Create validation split from training data
        if len(train_data) > 100:  # Only create validation if we have enough data
            val_size = min(0.2, 100 / len(train_data))  # At most 20% or 100 samples
            train_final, val_data = train_test_split(
                train_data, test_size=val_size, random_state=random_state
            )
        else:
            train_final = train_data
            val_data = None

        splits = {
            'train': train_final,
            'test': test_data,
            'full_dataset': dataset
        }

        if val_data is not None:
            splits['validation'] = val_data

        return splits

    def _is_target_column(self, col: str) -> bool:
        """Determine if a column is a target variable."""
        target_indicators = [
            'fine_imposed', 'fine_amount', 'fine_category',
            'severe_measures', 'remedy_only',
            'article5_violation', 'rights_violation',
            'enforcement_outcome'
        ]
        return any(indicator in col for indicator in target_indicators)

    def _export_datasets(self, splits: Dict[str, pd.DataFrame], output_dir: Path) -> None:
        """Export train/test/validation datasets."""
        for split_name, data in splits.items():
            # Export as CSV
            data.to_csv(output_dir / f'{split_name}.csv')

            # Export as Parquet if available
            try:
                data.to_parquet(output_dir / f'{split_name}.parquet', compression='snappy')
            except ImportError:
                pass

            # Export feature matrix and targets separately for convenience
            if split_name != 'full_dataset':
                feature_cols = [col for col in data.columns if not self._is_target_column(col)]
                target_cols = [col for col in data.columns if self._is_target_column(col)]

                if feature_cols:
                    data[feature_cols].to_csv(output_dir / f'{split_name}_features.csv')

                if target_cols:
                    data[target_cols].to_csv(output_dir / f'{split_name}_targets.csv')

    def _export_feature_metadata(self, dataset: pd.DataFrame, output_dir: Path) -> None:
        """Export comprehensive feature metadata."""
        feature_cols = [col for col in dataset.columns if not self._is_target_column(col)]
        target_cols = [col for col in dataset.columns if self._is_target_column(col)]

        metadata = {
            'dataset_info': {
                'total_samples': len(dataset),
                'total_features': len(feature_cols),
                'total_targets': len(target_cols),
                'feature_types': self._analyze_feature_types(dataset[feature_cols]),
                'target_distributions': self._analyze_target_distributions(dataset[target_cols]) if target_cols else {}
            },
            'feature_categories': {
                'geographic': [col for col in feature_cols if any(x in col for x in ['country', 'eu', 'eea'])],
                'temporal': [col for col in feature_cols if any(x in col for x in ['year', 'quarter', 'gdpr'])],
                'legal_complexity': [col for col in feature_cols if any(x in col for x in ['principles', 'measures', 'complexity'])],
                'violation_types': [col for col in feature_cols if any(x in col for x in ['article5', 'rights', 'security'])],
                'enforcement': [col for col in feature_cols if any(x in col for x in ['powers', 'enforcement', 'severity'])],
                'embeddings': [col for col in feature_cols if '_emb_' in col],
                'economic': [col for col in feature_cols if any(x in col for x in ['turnover', 'economic'])],
                'dpa_characteristics': [col for col in feature_cols if 'dpa_' in col],
                'sector': [col for col in feature_cols if any(x in col for x in ['isic', 'sector'])]
            },
            'target_variables': {
                'regression_targets': [col for col in target_cols if 'amount' in col or 'log' in col],
                'classification_targets': [col for col in target_cols if col not in [col for col in target_cols if 'amount' in col or 'log' in col]],
                'binary_targets': [col for col in target_cols if dataset[col].nunique() <= 2],
                'multiclass_targets': [col for col in target_cols if dataset[col].nunique() > 2]
            },
            'ml_recommendations': {
                'fine_prediction': {
                    'task_type': 'binary_classification_then_regression',
                    'features': 'Use all except economic features to avoid leakage',
                    'targets': 'fine_imposed (binary), fine_amount_log (regression)',
                    'notes': 'Two-stage model: first predict if fine imposed, then amount'
                },
                'violation_prediction': {
                    'task_type': 'multi_label_classification',
                    'features': 'Use legal_complexity, case characteristics, but not enforcement features',
                    'targets': 'article5_violation, rights_violation',
                    'notes': 'Predict violations from case characteristics'
                },
                'severity_prediction': {
                    'task_type': 'ordinal_classification',
                    'features': 'Use violation types and complexity indices',
                    'targets': 'enforcement_outcome',
                    'notes': 'Predict enforcement severity from violations'
                }
            },
            'legal_considerations': [
                'Avoid data leakage: enforcement features should not predict violations',
                'Account for country-specific legal traditions in evaluation',
                'Consider temporal effects: early decisions may differ from later ones',
                'Embeddings capture semantic similarity in legal reasoning',
                'Missing values in legal data may be meaningful (NOT_DISCUSSED vs NOT_MENTIONED)'
            ]
        }

        with open(output_dir / 'feature_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _export_preprocessing_artifacts(self, output_dir: Path) -> None:
        """Export preprocessing artifacts for reproducible ML pipelines."""
        if not JOBLIB_AVAILABLE:
            return

        artifacts_dir = output_dir / 'preprocessing_artifacts'
        artifacts_dir.mkdir(exist_ok=True)

        # Save scalers and encoders if they were used
        if hasattr(self, 'scaler') and hasattr(self.scaler, 'mean_'):
            joblib.dump(self.scaler, artifacts_dir / 'feature_scaler.joblib')

        if self.label_encoders:
            joblib.dump(self.label_encoders, artifacts_dir / 'label_encoders.joblib')

        # Create preprocessing pipeline script
        pipeline_script = '''
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_preprocessing_artifacts(artifacts_dir):
    """Load preprocessing artifacts."""
    scaler = joblib.load(artifacts_dir / 'feature_scaler.joblib')
    label_encoders = joblib.load(artifacts_dir / 'label_encoders.joblib')
    return scaler, label_encoders

def preprocess_new_data(df, scaler, label_encoders):
    """Apply preprocessing to new data."""
    # Apply same feature engineering as training data
    # (This would need to be customized based on actual preprocessing steps)

    # Scale numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Encode categorical features
    for col, encoder in label_encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])

    return df

# Example usage:
# scaler, encoders = load_preprocessing_artifacts('preprocessing_artifacts')
# new_data_processed = preprocess_new_data(new_data, scaler, encoders)
'''

        with open(artifacts_dir / 'preprocessing_pipeline.py', 'w') as f:
            f.write(pipeline_script)

    def _create_ml_templates(self, output_dir: Path) -> None:
        """Create ML analysis templates."""
        templates_dir = output_dir / 'ml_templates'
        templates_dir.mkdir(exist_ok=True)

        # Fine prediction template
        fine_prediction_template = '''
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

    print(f"\\nFine Amount Prediction (R²): {r2_score(y_fine_amount_test, y_pred_amount):.3f}")
    print(f"Fine Amount Prediction (RMSE): {np.sqrt(mean_squared_error(y_fine_amount_test, y_pred_amount)):.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance_classification': fine_classifier.feature_importances_,
        'importance_regression': fine_regressor.feature_importances_
    }).sort_values('importance_classification', ascending=False)

    print("\\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

# Legal interpretation notes
print("\\n" + "="*50)
print("LEGAL INTERPRETATION NOTES:")
print("="*50)
print("- This model predicts enforcement patterns, not legal correctness")
print("- Results should be validated against legal expertise")
print("- Consider country-specific legal traditions in interpretation")
print("- Temporal effects: early GDPR decisions may differ from current patterns")
'''

        with open(templates_dir / 'fine_prediction_template.py', 'w') as f:
            f.write(fine_prediction_template)

        # Violation analysis template
        violation_template = '''
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
    print(f"\\n{target.upper()} CLASSIFICATION REPORT:")
    print("="*50)
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance_article5': classifier.estimators_[0].feature_importances_,
    'importance_rights': classifier.estimators_[1].feature_importances_
}).sort_values('importance_article5', ascending=False)

print("\\nTOP 15 FEATURES FOR VIOLATION PREDICTION:")
print(feature_importance.head(15))

# Correlation analysis between violation types
violation_corr = y_test.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(violation_corr, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Between Violation Types')
plt.tight_layout()
plt.savefig('violation_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\\n" + "="*60)
print("LEGAL ANALYSIS INSIGHTS:")
print("="*60)
print("- High correlation between violation types suggests systemic compliance issues")
print("- Feature importance indicates key risk factors for violations")
print("- Results can inform proactive compliance strategies")
'''

        with open(templates_dir / 'violation_analysis_template.py', 'w') as f:
            f.write(violation_template)

    def _extract_violation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract Article 5 violation features."""
        features = pd.DataFrame({'decision_id': df['decision_id']})

        # Article 5 principles
        article5_cols = [col for col in df.columns if col.startswith('q31_violated_') and
                        not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]

        for col in article5_cols:
            principle = col.replace('q31_violated_', '')
            # Handle nullable boolean columns properly
            if col in df.columns:
                col_values = df[col]
                if col_values.dtype == 'boolean':
                    features[f'article5_{principle}'] = col_values.fillna(False).astype(int)
                else:
                    features[f'article5_{principle}'] = col_values.fillna(0).astype(int)

        return features.set_index('decision_id')

    def _extract_rights_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract data subject rights features."""
        features = pd.DataFrame({'decision_id': df['decision_id']})

        rights_cols = [col for col in df.columns if col.startswith('q57_rights_violated_') and
                      not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]

        for col in rights_cols:
            right = col.replace('q57_rights_violated_', '')
            # Handle nullable boolean columns properly
            if col in df.columns:
                col_values = df[col]
                if col_values.dtype == 'boolean':
                    features[f'rights_{right}'] = col_values.fillna(False).astype(int)
                else:
                    features[f'rights_{right}'] = col_values.fillna(0).astype(int)

        return features.set_index('decision_id')

    def _extract_enforcement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract enforcement power features."""
        features = pd.DataFrame({'decision_id': df['decision_id']})

        power_cols = [col for col in df.columns if col.startswith('q53_powers_') and
                     not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]

        for col in power_cols:
            power = col.replace('q53_powers_', '')
            # Handle nullable boolean columns properly
            if col in df.columns:
                col_values = df[col]
                if col_values.dtype == 'boolean':
                    features[f'power_{power}'] = col_values.fillna(False).astype(int)
                else:
                    features[f'power_{power}'] = col_values.fillna(0).astype(int)

        return features.set_index('decision_id')

    def _create_cross_border_indicator(self, df: pd.DataFrame) -> pd.Series:
        """Create cross-border case indicator."""
        cross_border = pd.Series(0, index=df.index)

        # Check Q49 and Q62 for cross-border indicators
        if 'raw_q49' in df.columns:
            cross_border |= df['raw_q49'].str.contains('YES_', na=False)
        if 'raw_q62' in df.columns:
            cross_border |= df['raw_q62'].str.contains('LEAD_|CONCERNED_|JOINT_', na=False)

        return cross_border.astype(int)

    def _create_severity_index(self, df: pd.DataFrame) -> pd.Series:
        """Create violation severity index."""
        severity = pd.Series(0, index=df.index)

        # Add points for different severity factors
        if 'n_principles_violated' in df.columns:
            severity += df['n_principles_violated'].fillna(0)

        if 'breach_case' in df.columns:
            severity += df['breach_case'].fillna(0) * 2  # Breach cases are more severe

        if 'severity_measures_present' in df.columns:
            severity += df['severity_measures_present'].fillna(0) * 3

        return severity

    def _create_dpa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create DPA-level features."""
        features = pd.DataFrame({'decision_id': df['decision_id']})

        # Encode DPA characteristics
        if 'dpa_name_canonical' in df.columns:
            # DPA enforcement frequency (decisions per DPA)
            dpa_counts = df['dpa_name_canonical'].value_counts()
            features['dpa_frequency'] = df['dpa_name_canonical'].map(dpa_counts)

            # DPA average fine amount
            dpa_avg_fine = df.groupby('dpa_name_canonical')['fine_eur'].mean()
            features['dpa_avg_fine'] = df['dpa_name_canonical'].map(dpa_avg_fine).fillna(0)

        return features.set_index('decision_id')

    def _create_sector_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create economic sector features."""
        features = pd.DataFrame({'decision_id': df['decision_id']})

        if 'isic_section' in df.columns:
            # One-hot encode ISIC sections
            sections = df['isic_section'].dropna().unique()
            for section in sections:
                features[f'sector_{section}'] = (df['isic_section'] == section).astype(int)

        return features.set_index('decision_id')

    def _create_article5_target(self, df: pd.DataFrame) -> pd.Series:
        """Create Article 5 violation target."""
        if 'n_principles_violated' in df.columns:
            return (df['n_principles_violated'] > 0).astype(int)
        return pd.Series(0, index=df.index)

    def _create_rights_target(self, df: pd.DataFrame) -> pd.Series:
        """Create rights violation target."""
        rights_cols = [col for col in df.columns if col.startswith('q57_rights_violated_') and
                      not col.endswith(('_coverage_status', '_known', '_unknown', '_status', '_exclusivity_conflict'))]
        if rights_cols:
            return (df[rights_cols].sum(axis=1) > 0).astype(int)
        return pd.Series(0, index=df.index)

    def _create_enforcement_outcome_target(self, df: pd.DataFrame) -> pd.Series:
        """Create multi-class enforcement outcome target."""
        # 0: No enforcement action, 1: Non-monetary measures only, 2: Fine imposed
        outcome = pd.Series(0, index=df.index)

        if 'severity_measures_present' in df.columns:
            outcome = outcome + df['severity_measures_present'].fillna(0)

        if 'fine_eur' in df.columns:
            outcome = outcome + (df['fine_eur'] > 0).astype(int)

        return outcome.clip(0, 2)  # Cap at 2

    def _analyze_feature_types(self, features_df: pd.DataFrame) -> Dict[str, int]:
        """Analyze feature types in the dataset."""
        return {
            'binary': len([col for col in features_df.columns if features_df[col].nunique() <= 2]),
            'categorical': len([col for col in features_df.columns if features_df[col].dtype == 'object']),
            'numeric': len([col for col in features_df.columns if features_df[col].dtype in ['int64', 'float64']]),
            'embeddings': len([col for col in features_df.columns if '_emb_' in col])
        }

    def _analyze_target_distributions(self, targets_df: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze target variable distributions."""
        distributions = {}
        for col in targets_df.columns:
            distributions[col] = {
                'type': 'binary' if targets_df[col].nunique() <= 2 else 'multiclass',
                'unique_values': int(targets_df[col].nunique()),
                'null_count': int(targets_df[col].isna().sum()),
                'value_counts': targets_df[col].value_counts().to_dict()
            }
        return distributions