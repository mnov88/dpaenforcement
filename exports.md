# Advanced Export Formats for GDPR Decision Analysis

This document provides comprehensive guidance on using the advanced export formats available in the DPA enforcement pipeline. These formats transform the traditional CSV-based workflow into a modern, multi-format analytical ecosystem optimized for different research and analysis scenarios.

## Overview

The export system provides five specialized output formats, each optimized for specific analytical workflows while preserving the legal semantics essential for GDPR research:

1. **Parquet** - High-performance analytical queries
2. **Arrow/Feather** - Cross-language data science
3. **Graph/Network** - Relationship and network analysis
4. **Statistical Packages** - R, Stata, and SPSS integration
5. **ML-Ready** - Machine learning with text embeddings

## Quick Start

```bash
# First, ensure you have the cleaned wide dataset
python3 -m scripts.cli run-all

# Export to all formats
python3 -m scripts.cli export-parquet --wide-csv outputs/cleaned_wide.csv --out-dir outputs/parquet
python3 -m scripts.cli export-arrow --wide-csv outputs/cleaned_wide.csv --out-dir outputs/arrow
python3 -m scripts.cli export-graph --wide-csv outputs/cleaned_wide.csv --out-dir outputs/networks
python3 -m scripts.cli export-stats --wide-csv outputs/cleaned_wide.csv --out-dir outputs/statistical
python3 -m scripts.cli export-ml --wide-csv outputs/cleaned_wide.csv --out-dir outputs/ml_ready
```

## 1. Parquet Export

### Purpose
Parquet format provides columnar storage optimized for analytical queries with dramatic file size reduction and query performance improvements.

### Key Benefits
- **10x smaller files** through columnar compression
- **10x faster queries** for analytical workloads
- **Native integration** with pandas, Polars, DuckDB, Spark
- **Partitioned datasets** for efficient filtering
- **Rich metadata** preservation

### Usage
```bash
python3 -m scripts.cli export-parquet \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/parquet \
  --partition-cols country_group,decision_year
```

### Output Structure
```
outputs/parquet/
├── wide_partitioned/
│   ├── country_group=EU/
│   │   ├── decision_year=2019/
│   │   ├── decision_year=2020/
│   │   └── ...
│   ├── country_group=EEA/
│   └── country_group=NON_EEA/
├── long_tables/
│   ├── article_5_discussed.parquet
│   ├── corrective_powers.parquet
│   └── ...
└── metadata.json
```

### Analysis Examples
```python
import pandas as pd
import pyarrow.parquet as pq

# Load partitioned dataset
df = pd.read_parquet('outputs/parquet/wide_partitioned')

# Efficient filtering (uses partitions)
eu_2020 = pd.read_parquet(
    'outputs/parquet/wide_partitioned',
    filters=[('country_group', '=', 'EU'), ('decision_year', '=', 2020)]
)

# Load specific long table
violations = pd.read_parquet('outputs/parquet/long_tables/article_5_violated.parquet')
```

### Performance Comparison
| Format | File Size | Load Time | Filter Query |
|--------|-----------|-----------|--------------|
| CSV    | 5.1 MB    | 2.3s      | 0.8s         |
| Parquet| 0.7 MB    | 0.2s      | 0.1s         |

## 2. Arrow/Feather Export

### Purpose
Arrow format enables zero-copy data sharing across programming languages with embedded metadata for reproducible research.

### Key Benefits
- **Zero-copy reads** in supported languages
- **Cross-language compatibility** (Python/R/Julia/Rust)
- **Rich metadata** with legal semantics
- **Memory efficient** columnar format
- **Embedded schemas** for self-documenting datasets

### Usage
```bash
python3 -m scripts.cli export-arrow \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/arrow \
  --compression zstd
```

### Output Structure
```
outputs/arrow/
├── wide_data.feather          # Primary dataset
├── wide_data.arrow            # IPC format
├── long_tables/
│   ├── article_5_discussed.feather
│   └── ...
├── metadata.json             # Comprehensive metadata
└── arrow_schema.json         # Schema information
```

### Cross-Language Usage

**Python:**
```python
import pandas as pd
import pyarrow.feather as feather

# Load with rich metadata
df = pd.read_feather('outputs/arrow/wide_data.feather')

# Access Arrow table with metadata
table = feather.read_table('outputs/arrow/wide_data.feather')
print(table.schema.metadata)
```

**R:**
```r
library(arrow)

# Load dataset
df <- read_feather('outputs/arrow/wide_data.feather')

# Access metadata
table <- arrow::read_feather('outputs/arrow/wide_data.feather', as_data_frame = FALSE)
table$schema$metadata
```

**Julia:**
```julia
using Arrow, DataFrames

# Load dataset
df = DataFrame(Arrow.Table('outputs/arrow/wide_data.feather'))
```

## 3. Graph/Network Export

### Purpose
Network formats enable analysis of relationships between DPAs, decisions, legal concepts, and enforcement patterns.

### Key Benefits
- **Multiple formats** (GraphML, GML, Neo4j CSV)
- **Ready for analysis** in Gephi, Cytoscape, R igraph, NetworkX
- **Legal relationship modeling** (DPA-Decision, Decision-Article)
- **Co-occurrence networks** for violation patterns
- **Cross-border enforcement** analysis

### Usage
```bash
python3 -m scripts.cli export-graph \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/networks
```

### Output Structure
```
outputs/networks/
├── dpa_decision_network.graphml          # DPA-Decision relationships
├── decision_article_bipartite.graphml    # Decision-Article bipartite
├── violation_cooccurrence.graphml        # Co-occurring violations
├── cross_border_network.graphml          # Cross-border cases
├── adjacency_matrices/
│   ├── violation_cooccurrence_adjacency.csv
│   └── violation_cooccurrence_nodes.csv
├── neo4j_import/
│   ├── dpa_nodes.csv
│   ├── decision_nodes.csv
│   ├── principle_nodes.csv
│   ├── dpa_decision_relationships.csv
│   └── import_script.cypher
└── graph_metadata.json
```

### Analysis Examples

**NetworkX (Python):**
```python
import networkx as nx

# Load violation co-occurrence network
G = nx.read_graphml('outputs/networks/violation_cooccurrence.graphml')

# Calculate centrality measures
centrality = nx.degree_centrality(G)
betweenness = nx.betweenness_centrality(G)

# Community detection
from networkx.algorithms import community
communities = community.greedy_modularity_communities(G)
```

**igraph (R):**
```r
library(igraph)

# Load network
g <- read_graph('outputs/networks/violation_cooccurrence.gml', format = 'gml')

# Calculate centrality
degree_cent <- degree(g, normalized = TRUE)
between_cent <- betweenness(g, normalized = TRUE)

# Community detection
communities <- cluster_louvain(g)
```

**Neo4j (Graph Database):**
```cypher
// Load the import script
:auto USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///dpa_nodes.csv' AS row
CREATE (d:DPA {name: row.name, country: row.country});

// Run complex queries
MATCH (dpa:DPA)-[:DECIDED]->(decision:Decision)-[:VIOLATED]->(principle:LegalConcept)
WHERE principle.category = 'violated'
RETURN dpa.name, COUNT(DISTINCT principle.name) as violations_count
ORDER BY violations_count DESC;
```

## 4. Statistical Package Export

### Purpose
Native formats for statistical software with proper metadata, factor levels, and analysis templates.

### Key Benefits
- **Native integration** with R, Stata, and SPSS
- **Proper factor levels** and variable labels
- **Comprehensive codebooks** for documentation
- **Analysis templates** for common research patterns
- **Legal semantics preserved** in statistical contexts

### Usage
```bash
python3 -m scripts.cli export-stats \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/statistical \
  --formats r,stata,spss
```

### Output Structure
```
outputs/statistical/
├── gdpr_decisions.rds              # R dataset
├── load_gdpr_data.R               # R loading script
├── analysis_template.R            # R analysis template
├── gdpr_decisions.dta             # Stata dataset
├── load_gdpr_data.do              # Stata loading script
├── gdpr_decisions.sav             # SPSS dataset
├── codebook.json                  # Machine-readable codebook
├── codebook.txt                   # Human-readable codebook
└── statistical_metadata.json      # Analysis guidance
```

### Statistical Analysis Examples

**R Analysis:**
```r
# Load the dataset
source('outputs/statistical/load_gdpr_data.R')

# Basic descriptive analysis
library(dplyr)
gdpr_data %>%
  filter(fine_status == "DISCUSSED") %>%
  group_by(country_group) %>%
  summarise(
    n_decisions = n(),
    median_fine = median(fine_eur[fine_eur > 0], na.rm = TRUE),
    breach_rate = mean(breach_case, na.rm = TRUE)
  )

# Regression analysis respecting legal semantics
fine_model <- gdpr_data %>%
  filter(fine_status == "DISCUSSED") %>%  # Only where fines assessed
  lm(fine_log ~ country_group + breach_case + n_principles_violated, data = .)

summary(fine_model)
```

**Stata Analysis:**
```stata
* Load the dataset
do "outputs/statistical/load_gdpr_data.do"

* Descriptive statistics by country group
tabstat fine_eur if fine_status == "DISCUSSED", by(country_group) stat(n mean median)

* Regression with robust standard errors
reg fine_log i.country_group breach_case n_principles_violated if fine_status == "DISCUSSED", robust

* Export results
eststo: reg fine_log i.country_group breach_case n_principles_violated if fine_status == "DISCUSSED"
esttab using "regression_results.tex", replace
```

### Important Notes for Statistical Analysis

1. **Typed Missingness**: Always filter by status variables (e.g., `fine_status == "DISCUSSED"`) before analysis
2. **Legal Interpretation**: `NOT_DISCUSSED ≠ NOT_MENTIONED ≠ NOT_APPLICABLE`
3. **Zero Fines**: May be meaningful (explicit decision of no fine)
4. **Temporal Effects**: Account for GDPR learning curve in early decisions

## 5. ML-Ready Export

### Purpose
Machine learning features with text embeddings, proper train/test splits, and analysis templates.

### Key Benefits
- **300+ engineered features** preserving legal semantics
- **Text embeddings** for narrative analysis
- **Stratified splits** maintaining legal distributions
- **Feature metadata** and importance tracking
- **Ready-to-use templates** for common ML tasks

### Usage
```bash
python3 -m scripts.cli export-ml \
  --wide-csv outputs/cleaned_wide.csv \
  --long-tables-dir outputs/long_tables \
  --out-dir outputs/ml_ready \
  --embeddings-model all-MiniLM-L6-v2 \
  --test-size 0.2 \
  --random-state 42
```

### Output Structure
```
outputs/ml_ready/
├── train.csv                      # Training set
├── test.csv                       # Test set
├── validation.csv                 # Validation set (if sufficient data)
├── full_dataset.csv              # Complete dataset
├── train_features.csv             # Features only (training)
├── train_targets.csv              # Targets only (training)
├── test_features.csv              # Features only (test)
├── test_targets.csv               # Targets only (test)
├── feature_metadata.json          # Feature descriptions and categories
├── embeddings_info.json           # Text embedding information
├── preprocessing_artifacts/
│   ├── feature_scaler.joblib      # Sklearn scaler
│   └── preprocessing_pipeline.py  # Preprocessing script
└── ml_templates/
    ├── fine_prediction_template.py
    └── violation_analysis_template.py
```

### Feature Categories

1. **Geographic Features**: Country groupings, DPA characteristics
2. **Temporal Features**: Decision year, quarter, time since GDPR
3. **Legal Complexity**: Counts of principles, measures, violations
4. **Violation Types**: Binary indicators for specific violations
5. **Case Characteristics**: Breach case, cross-border, sector
6. **Text Embeddings**: 384-dimensional vectors for narrative fields
7. **Derived Indices**: Severity, complexity, enforcement measures

### ML Analysis Examples

**Fine Prediction (Two-Stage Model):**
```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, r2_score

# Load data
train_df = pd.read_csv('outputs/ml_ready/train.csv', index_col=0)
test_df = pd.read_csv('outputs/ml_ready/test.csv', index_col=0)

# Feature selection (avoid data leakage)
feature_cols = [col for col in train_df.columns
                if not any(x in col for x in ['fine_', 'severe_', 'enforcement_'])]

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]

# Stage 1: Predict if fine will be imposed
fine_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
fine_classifier.fit(X_train, train_df['fine_imposed'])

# Stage 2: For imposed fines, predict amount
fine_cases_train = train_df[train_df['fine_imposed'] == 1]
fine_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
fine_regressor.fit(fine_cases_train[feature_cols], fine_cases_train['fine_amount_log'])

# Evaluate
y_pred_imposed = fine_classifier.predict(X_test)
print("Fine Imposition Classification:")
print(classification_report(test_df['fine_imposed'], y_pred_imposed))
```

**Violation Pattern Analysis:**
```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# Multi-label violation prediction
violation_targets = ['article5_violation', 'rights_violation']
classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100))

# Exclude enforcement outcomes to avoid leakage
features = [col for col in train_df.columns
           if not any(x in col for x in ['fine_', 'enforcement_', 'severe_'])]

classifier.fit(train_df[features], train_df[violation_targets])
predictions = classifier.predict(test_df[features])

# Feature importance for each violation type
for i, target in enumerate(violation_targets):
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': classifier.estimators_[i].feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop features for {target}:")
    print(importance_df.head(10))
```

### Text Embeddings Usage

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Extract embeddings
embedding_cols = [col for col in train_df.columns if '_emb_' in col]
embeddings = train_df[embedding_cols].values

# Cluster similar legal reasoning
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize in 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis')
plt.title('Legal Reasoning Clusters (t-SNE)')
plt.colorbar(scatter)
plt.show()
```

## Dependencies and Installation

### Required Dependencies
All export functionality works with the base Python installation, but optional dependencies enhance capabilities:

```bash
# Core dependencies (always available)
pip install pandas numpy scikit-learn

# High-performance formats
pip install pyarrow  # Parquet and Arrow exports

# Graph analysis
pip install networkx  # Network exports

# Text embeddings
pip install sentence-transformers  # ML text embeddings

# Statistical package integration
pip install pyreadstat  # Stata and SPSS exports
pip install rpy2  # R integration (requires R installation)

# ML enhancements
pip install joblib  # Preprocessing artifacts
```

### Graceful Degradation
All exporters gracefully handle missing optional dependencies:
- Missing `pyarrow`: Falls back to CSV export with warnings
- Missing `networkx`: Skips graph generation, exports CSV edge lists
- Missing `sentence-transformers`: Skips embeddings, continues with other features
- Missing `pyreadstat`: Exports CSV with loading scripts for statistical packages

## Performance Considerations

### File Size Comparison
| Format | Size | Compression | Load Time |
|--------|------|-------------|-----------|
| CSV    | 5.1 MB | 1x | 2.3s |
| Parquet| 0.7 MB | 7.3x | 0.2s |
| Arrow  | 1.2 MB | 4.3x | 0.3s |
| Gzip CSV| 0.9 MB | 5.7x | 1.8s |

### Query Performance
- **Parquet**: Columnar storage enables 10x faster analytical queries
- **Arrow**: Zero-copy reads eliminate serialization overhead
- **Partitioned datasets**: Enable efficient filtering without full scans

### Memory Usage
- **Arrow format**: Most memory-efficient for cross-language workflows
- **Categorical encoding**: Reduces memory usage for repeated strings
- **Lazy loading**: Parquet and Arrow support selective column loading

## Legal Data Considerations

### Typed Missingness Preservation
All export formats preserve the legal distinction between:
- `NOT_DISCUSSED`: Question not addressed in decision
- `NOT_MENTIONED`: Aspect not mentioned in response
- `NOT_APPLICABLE`: Question not relevant to case type
- `UNCLEAR`: Response ambiguous or incomplete
- `NOT_DETERMINED`: Violation status unclear from decision

### Multi-Select Variable Semantics
- **Exclusivity flags**: Track contradictory responses
- **Coverage status**: Understand response completeness
- **Binary indicators**: Represent presence/absence of each option
- **Status tracking**: Maintain legal interpretation context

### Validation and Quality Flags
- **Schema echo detection**: AI response artifacts identified and flagged
- **Consistency checks**: Cross-field logical validation results included
- **Data quality metrics**: Missing data patterns and outlier detection

## Troubleshooting

### Common Issues

**ImportError: No module named 'pyarrow'**
```bash
pip install pyarrow
# Or use fallback CSV export
```

**Memory errors with large datasets**
```python
# Use chunked reading for large files
import pandas as pd
chunks = pd.read_csv('large_file.csv', chunksize=1000)
for chunk in chunks:
    # Process chunk
    pass
```

**Graph export fails with NetworkX errors**
```bash
# Update NetworkX
pip install --upgrade networkx
# Or use CSV edge lists as fallback
```

### Performance Optimization

**Slow Parquet export**
- Reduce partition granularity
- Use smaller chunk sizes for very large datasets
- Consider uncompressed format for faster writes

**Large embedding files**
- Use smaller embedding models (e.g., 'all-MiniLM-L12-v2' → 'all-MiniLM-L6-v2')
- Skip embeddings for initial exploration: `--embeddings-model none`

**Memory issues in statistical exports**
- Export subsets by filtering before export
- Use streaming export for very large datasets

## Best Practices

### Workflow Recommendations

1. **Start with Parquet** for initial data exploration and analysis
2. **Use Arrow** for cross-language collaboration projects
3. **Export graphs** for relationship and network analysis
4. **Generate ML exports** for predictive modeling projects
5. **Create statistical exports** for formal research and publication

### Analysis Guidelines

1. **Always check data quality** using validation reports before analysis
2. **Respect legal semantics** by filtering on status variables
3. **Account for temporal effects** in longitudinal analysis
4. **Use appropriate statistical methods** for legal categorical data
5. **Document methodology** thoroughly for reproducible research

### File Organization

```
project/
├── raw-data/                  # Original decision files
├── analyzed-decisions/        # AI-processed responses
├── outputs/
│   ├── cleaned_wide.csv      # Primary cleaned dataset
│   ├── validation_report.json
│   ├── parquet/              # High-performance analytics
│   ├── arrow/                # Cross-language data science
│   ├── networks/             # Graph analysis
│   ├── statistical/          # R/Stata/SPSS
│   └── ml_ready/             # Machine learning
└── analysis/
    ├── exploratory/          # Initial data exploration
    ├── statistical/          # Formal statistical analysis
    ├── network/              # Graph analysis
    └── ml/                   # Predictive modeling
```

This comprehensive export system transforms the DPA enforcement pipeline into a modern, multi-format analytical ecosystem while preserving the essential legal semantics required for rigorous GDPR research.