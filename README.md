# Credit Card Default Prediction - Machine Learning Project Report

## Introduction

 This project implements a comprehensive machine learning pipeline to predict credit card payment defaults using the UCI "Default of Credit Card Clients" dataset. The dataset contains 30,000 instances from a Taiwan-based credit card issuer (2005) with 23 features spanning demographic information, credit data, payment history, and bill statements from April to September 2005.
        
 The business objective is to enable financial institutions to assess credit risk and predict the likelihood of customer defaults, facilitating better credit limit decisions and risk management strategies. This is a binary classification problem where the target variable indicates whether a client will default on their credit card payment (1 = Yes, 0 = No).

## Analysis Process and Methodology

### Week 1: Data Collection and Preprocessing

#### Data Loading and Exploration

  The dataset was loaded using pandas with an initial shape of 30,000 rows × 25 columns. The ID column was dropped as it serves only as a unique identifier with no predictive value.

#### Missing Values and Duplicates

  The dataset showed zero missing values across all features, indicating high data quality. However, 35 duplicate records were identified and removed, resulting in 29,965 unique instances.

#### Outlier Detection and Treatment

 Using the IQR (Interquartile Range) method, outliers were detected across numerical features. Key findings included:
    
•	LIMIT_BAL: 167 outliers (0.56%)

•	AGE: 272 outliers (0.91%)

•	Bill amount features (BILL_AMT1-6): 7-9% outliers each

•	Payment amount features (PAY_AMT1-6): 8-10% outliers each

Outliers were handled using clipping to the IQR bounds rather than removal to preserve data.

#### Skewness Correction

Original skewness analysis revealed that most continuous features were moderately to highly skewed:

•	LIMIT_BAL: 0.905 (moderately skewed)

•	Bill amounts: 1.18-1.20 (highly skewed)

•	Payment amounts: 1.03-1.21 (highly skewed)

Various transformation methods (log, square root, Yeo-Johnson, Box-Cox) were applied, with square root transformation proving most effective for reducing skewness in most features.

### Week 2: Exploratory Data Analysis

#### Target Variable Distribution

The dataset exhibits significant class imbalance with a ratio of 3.52:1:

•	Non-defaulters (0): 23,335 instances (77.9%)

•	Defaulters (1): 6,630 instances (22.1%)[1]

| Attribute | Value |
| :-------- | :---- |
| Total Instances | 30,000 |
| Features | 23 |
| Missing Values | 0 |
| Duplicates Removed | 35 |
| Final Dataset Size | 29,965 |
| Default Rate | 22.1% |
| Non-Default Rate | 77.9% |

#### Class Imbalance Handling

To address the moderate-to-high class imbalance, a hybrid resampling approach was implemented using SMOTE (Synthetic Minority Over-sampling Technique) and RandomUnderSampler. This reduced the imbalance ratio from 3.52 to 1.43:

•	Resampled non-defaulters: 16,667

•	Resampled defaulters: 11,667

##### Dataset Overview

| Metric | Before Preprocessing | After Preprocessing |
| :----- | :------------------: | :-----------------: |
| Total Records | 30,000 | 29,965 |
| Outliers | Present | Clipped |
| Skewness | High (1.0+) | Corrected (<0.5) |
| Class Balance | 3.52:1 | 1.43:1 |

#### Univariate Analysis

Visualizations revealed important patterns:

•	Credit limits are right-skewed with concentration in lower ranges

•	Age distribution shows typical working-age concentration

•	Payment history features show variations in repayment behavior

#### Bivariate and Multivariate Analysis

Key insights from correlation and relationship analysis:

•	Customers with higher credit limits tend to have lower default rates

•	Payment history features (PAY_0 through PAY_6) show strong correlations with default status

•	Gender shows marginal differences in default rates

•	Education and marriage status combinations influence default probability

### Week 3: Feature Engineering and Model Building

#### Feature Preparation

All features were numerical with no categorical variables requiring encoding. Standard scaling was applied to normalize feature ranges using StandardScaler.

#### Train-Test Split

  Data was split using an 80-20 ratio with stratification to maintain class distribution:
•	Training set: 23,972 samples
•	Test set: 5,993 samples

#### Model Training and Evaluation

Six classification algorithms were trained and evaluated:

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :-------------------- | :------------: | :-----------: | :-------: | :----: | :------: | :-----: |
| Gradient Boosting     | 0.8279         | **0.8203**    | 0.8034    | 0.8203 | 0.7987   | **0.7729** |
| Support Vector Machine| 0.8263         | 0.8161        | 0.7978    | 0.8161 | 0.7918   | 0.7275 |
| Logistic Regression   | 0.8074         | 0.8114        | 0.7969    | 0.8114 | 0.7728   | 0.7232 |
| Random Forest         | 0.9994         | 0.8084        | 0.7879    | 0.8084 | 0.7883   | 0.7605 |
| K-Nearest Neighbors   | 0.8425         | 0.7884        | 0.7630    | 0.7884 | 0.7681   | 0.6957 |
| Decision Tree         | 0.9994         | 0.7272        | 0.7306    | 0.7272 | 0.7289   | 0.6112 |

Gradient Boosting emerged as the best-performing model with a test accuracy of 82.03% and ROC AUC of 0.7729.[1]

##### Model Performance Results

| Model | Accuracy | ROC AUC | Status |
| :---- | :------: | :-----: | :----- |
| Gradient Boosting | 82.03% | 0.7729 | ✅ Best Model |
| SVM | 81.61% | 0.7275 | ✓ Good |
| Logistic Regression | 81.14% | 0.7232 | ✓ Good |

### Week 4: Optimization and Finalization

#### Hyperparameter Tuning

 The Gradient Boosting model underwent hyperparameter optimization. The final optimized parameters were:

•	n_estimators: 100

•	learning_rate: 0.1[1]

#### Cross-Validation Performance

 The tuned model achieved a mean cross-validation ROC AUC score of 0.7820 with low standard deviation (0.0048), indicating stable and reliable performance.[1]
Feature Importance Analysis
The model's feature importance analysis revealed the top predictors:

| Rank | Feature | Importance Score | Description |
| :--: | :------ | :--------------: | :---------- |
| 1    | PAY_0   | 0.628            | Repayment status in September (Most Recent) |
| 2    | PAY_2   | 0.077            | Repayment status in August |
| 3    | BILL_AMT1 | 0.040          | Bill statement in September |
| 4    | PAY_3   | 0.037            | Repayment status in July |
| 5    | LIMIT_BAL | 0.030          | Credit limit amount |

The PAY_0 feature dominates with 62.8% importance, indicating that the most recent payment status is by far the strongest predictor of default.

##### Conclusions and Inferences

###### Key Findings

1.	Payment History is Critical: The analysis conclusively demonstrates that recent payment behavior (PAY_0) is the most powerful indicator of future default risk, accounting for nearly two-thirds of the model's decision-making. This aligns with financial industry intuition that past behavior predicts future performance.

2.	Model Performance: The final Gradient Boosting model achieves 82.03% test accuracy with a ROC AUC of 0.77, providing reliable predictions for credit risk assessment. The model maintains good balance between precision (80.34%) and recall (82.03%).

3.	Class Imbalance Management: The hybrid SMOTE and undersampling approach successfully addressed the 3.52:1 class imbalance, reducing it to 1.43:1 without significantly compromising model performance.

4.	Feature Relationships: Bill amounts and credit limits show moderate importance, suggesting that the absolute monetary values matter less than payment behavior patterns.

###### Business Implications

•	Risk Assessment: The model can identify high-risk customers with 82% accuracy, enabling proactive credit limit adjustments

•	Cost Reduction: By predicting defaults early, financial institutions can minimize losses through early intervention strategies

•	Credit Decisioning: The feature importance insights guide which customer attributes to prioritize during credit evaluations

###### Model Deployment
The trained pipeline has been saved as tuned_gradient_boosting_pipeline.joblib for production deployment. The pipeline encapsulates all preprocessing and modeling steps, making it ready for real-time credit risk assessment.

###### Limitations and Future Work

 Current Limitations:

•	Dataset is from 2005; modern credit behavior patterns may differ

•	Model performance could be enhanced for the minority class (defaulters)

•	Limited to Taiwanese credit market; generalization to other markets uncertain

###### Recommended Future Enhancements:

•	Explore advanced ensemble methods (XGBoost, LightGBM, CatBoost)

•	Implement cost-sensitive learning to account for the different costs of false positives vs. false negatives

•	Engineer additional features such as payment-to-bill ratios, trend indicators, and seasonal patterns

•	Investigate deep learning approaches for complex pattern recognition

•	Incorporate external economic indicators and macroeconomic variables

•	Deploy model monitoring to track prediction drift over time

##### Final Performance Summary

Tuned Gradient Boosting Classifier:

| Metric | Value |
| :-------------------------- | :----: |
| Test Accuracy | 82.03% |
| Precision | 80.34% |
| Recall | 82.03% |
| F1-Score | 79.87% |
| ROC AUC | 77.29% |
| Cross-Validation ROC AUC | 78.20% (±0.48%) |

 The model demonstrates strong predictive capability and can serve as a reliable tool for credit risk assessment in financial institutions, with the understanding that continuous monitoring and periodic retraining will be necessary to maintain performance in changing economic conditions.
