# Phase 2: Breach Profiling and Clustering Results

## Clustering Diagnostics

- Method: PCA_KMeans
- Number of clusters: 6
- Silhouette score: 0.220
- PCA explained variance: 0.802
- PCA components: 11

## Cluster Profiles

### Cluster_0: Country No, Vulnerable Subjects, Channel Breach Notification
**Size**: 15 cases
**Description**: Characterized by country no, vulnerable subjects, channel breach notification

**Key Characteristics**:
- Breach Type Organizational Failure: 0.600
- Breach Type Technical Failure: 0.267
- Special Article9: 0.533
- Vulnerable Subjects: 0.800
- Channel Breach Notification: 0.733
- Country No: 1.000
- Timing Compliant: 0.400

**Expected Outcomes**:
- Fine Probability: 0.867
- Avg Log Fine: 11.257
- Severity Index: 1.800

### Cluster_1: Channel Breach Notification, Vulnerable Subjects, Subjects Notified
**Size**: 72 cases
**Description**: Characterized by channel breach notification, vulnerable subjects, subjects notified

**Key Characteristics**:
- Breach Type Cyber Attack: 0.486
- Special Article9: 0.569
- Special Article10: 0.069
- Vulnerable Subjects: 0.639
- Channel Breach Notification: 0.958
- Country It: 0.208
- Timing Compliant: 0.528
- Subjects Notified: 0.583

**Expected Outcomes**:
- Fine Probability: 0.722
- Avg Log Fine: 8.687
- Severity Index: 1.681

### Cluster_2: Channel Complaint, Breach Type Human Error, Country Es
**Size**: 63 cases
**Description**: Characterized by channel complaint, breach type human error, country es

**Key Characteristics**:
- Breach Type Human Error: 0.619
- Channel Complaint: 0.921
- Country Es: 0.619
- Country Gr: 0.095

**Expected Outcomes**:
- Fine Probability: 0.603
- Avg Log Fine: 6.614
- Severity Index: 1.492

### Cluster_3: Breach Type Human Error, Channel Ex Officio Dpa Initiative, Country Pl
**Size**: 32 cases
**Description**: Characterized by breach type human error, channel ex officio dpa initiative, country pl

**Key Characteristics**:
- Breach Type Human Error: 0.688
- Special Article9: 0.531
- Channel Ex Officio Dpa Initiative: 0.656
- Country Pl: 0.656

**Expected Outcomes**:
- Fine Probability: 0.875
- Avg Log Fine: 10.223
- Severity Index: 1.844

### Cluster_4: Remedial Actions, Country Fr, Breach Type Organizational Failure
**Size**: 8 cases
**Description**: Characterized by remedial actions, country fr, breach type organizational failure

**Key Characteristics**:
- Breach Type Organizational Failure: 0.625
- Breach Type Technical Failure: 0.500
- Remedial Actions: 1.000
- Channel Ex Officio Dpa Initiative: 0.250
- Country Fr: 1.000
- Timing Compliant: 0.375

**Expected Outcomes**:
- Fine Probability: 0.875
- Avg Log Fine: 10.683
- Severity Index: 1.750

### Cluster_5: Country Ro, Channel Breach Notification, Breach Type Organizational Failure
**Size**: 18 cases
**Description**: Characterized by country ro, channel breach notification, breach type organizational failure

**Key Characteristics**:
- Breach Type Organizational Failure: 0.389
- Channel Breach Notification: 0.667
- Country Ro: 0.889

**Expected Outcomes**:
- Fine Probability: 1.000
- Avg Log Fine: 9.061
- Severity Index: 2.000

## Notes

- Clusters are derived using PCA for dimensionality reduction followed by K-means
- Optimal cluster number selected based on silhouette score
- Characteristics shown are features >20% above overall average
- Expected outcomes are cluster-specific means for key enforcement variables