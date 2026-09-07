# Telco Customer Churn Analysis — Internship Project Report

## 1. Executive Summary
The dataset contains **7,043 customers**, including **1,869 churned customers** and **5,174 retained customers**. The overall churn rate is **26.5%** and retention rate is **73.5%**. Average tenure is **32.4 months** and average monthly charges are **$64.76**.

The analysis shows a pronounced early-lifecycle churn pattern: churn is **52.9%** among customers with 0–6 months of tenure and falls to **9.5%** among customers with 49–72 months. Month-to-month customers show **42.7%** churn compared with **11.3%** for one-year and **2.8%** for two-year contracts.

A logistic-regression model achieved **0.845 ROC-AUC** and **0.805 recall** on a stratified 20% holdout set. The model is used as an interpretable risk-scoring framework, not as proof of causal churn drivers.

## 2. Business Problem
Subscription businesses lose recurring revenue when customers churn. Management needs to know:
- where in the customer lifecycle churn is concentrated;
- which contract, service and billing segments are associated with higher churn;
- whether observed differences are statistically significant;
- which customers can be prioritized for retention outreach; and
- what additional data is needed to improve intervention quality.

## 3. Data Preparation
The workflow:
1. audits data types, missing values and duplicates;
2. normalizes text fields;
3. converts `TotalCharges` to numeric;
4. handles blank total charges for zero-tenure records;
5. removes duplicate customer IDs;
6. creates `ChurnFlag`;
7. creates tenure-based lifecycle bands; and
8. creates historical revenue/value proxies for descriptive analysis.

## 4. Exploratory and Lifecycle Analysis
The project evaluates overall churn/retention, tenure, contract, internet service, payment method, billing and service attributes.

### Lifecycle insight
The declining churn rate across tenure bands indicates that the early customer lifecycle is the highest-priority intervention window. Because the dataset has no signup date, these are **tenure-based lifecycle cohorts**, not calendar-month acquisition cohorts.

### Contract insight
Month-to-month customers have much higher observed churn than one-year and two-year customers. This is an association in the historical data; it should not be interpreted as a causal effect without experimental or longitudinal evidence.

## 5. Revenue Analysis
The project calculates **$139,130.85 of monthly charges associated with customers who historically churned**. This is a historical exposure metric. A true forward-looking revenue-at-risk metric should be calculated from currently active customers and their predicted churn probability.

## 6. Statistical Testing
### Chi-square tests
Chi-square tests are used for categorical variables such as contract, internet service, payment method, billing and service attributes. Because multiple categorical hypotheses are tested, a Benjamini–Hochberg false-discovery-rate correction is also reported.

### Welch's t-tests
Welch's t-tests compare churned and retained groups for tenure, monthly charges and total charges without assuming equal variances.

## 7. Predictive Modeling
Logistic Regression was selected because it is interpretable and produces probability scores. Preprocessing is handled in a scikit-learn pipeline with imputation, standardization and one-hot encoding.

### Holdout performance
| Metric | Result |
|---|---:|
| Accuracy | 73.9% |
| Precision | 50.5% |
| Recall | 80.5% |
| F1 | 62.1% |
| ROC-AUC | 84.5% |

The model is first evaluated on unseen test data. It is then refit on all available historical records before final portfolio risk scoring. Historical aggregate value variables are excluded from the predictive feature set to reduce redundancy and keep the intervention model closer to scoring-time information.

## 8. Retention Strategy
1. **Early-lifecycle program:** strengthen onboarding, activation and first-value milestones in the first 6–12 months.
2. **High-risk/high-value outreach:** prioritize active customers with high model scores and meaningful monthly charges.
3. **Contract conversion tests:** test appropriate incentives for moving suitable month-to-month customers to longer commitments.
4. **Support/service intervention:** investigate high-churn segments linked to service and support attributes.
5. **Reason capture:** add cancellation reason, exit survey, support-ticket and satisfaction fields.
6. **Data enrichment:** add product usage, engagement, acquisition channel and payment behavior.
7. **Experimentation:** measure incremental retention through controlled tests rather than assuming correlation is causation.

## 9. Limitations
- No signup date → no true calendar-month cohort analysis.
- No explicit churn-reason field → actual reasons cannot be claimed.
- No region field → geographic analysis is unavailable.
- Historical dataset → model performance may not generalize to current customers.
- Risk thresholds are illustrative and should be tuned to business capacity and intervention economics.

## 10. Conclusion
The project converts a standard Telco churn dataset into a complete analytics workflow: data quality, lifecycle segmentation, statistical inference, interpretable prediction, risk prioritization and management recommendations. The strongest practical message is to focus retention resources on the early lifecycle and high-risk customer segments while collecting richer behavioral data to improve future targeting.
