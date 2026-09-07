# Internship Presentation — Customer Churn & Retention Analytics

## Slide 1 — Title
**Telco Customer Churn Analysis & Retention Strategy**

## Slide 2 — Business Problem
- Churn reduces recurring revenue and customer lifetime value.
- Goal: identify high-risk customer segments and prioritize retention actions.

## Slide 3 — Dataset & Data Quality
- 7,043 customers; 21 original variables.
- `TotalCharges` converted to numeric.
- Missing values and duplicate customer IDs audited.
- Created `ChurnFlag` and tenure-based lifecycle bands.

## Slide 4 — Executive KPIs
- Churn: **26.5%**
- Retention: **73.5%**
- Average tenure: **32.4 months**
- Average monthly charges: **$64.76**
- Historical monthly charges associated with churned customers: **$139,130.85**

## Slide 5 — Lifecycle Analysis
Use `outputs/charts/02_churn_by_tenure.png`.
- 0–6 months: **52.9% churn**
- 49–72 months: **9.5% churn**
- Message: the first months are the critical retention window.

## Slide 6 — Contract & Segment Analysis
Use `outputs/charts/03_churn_by_contract.png`.
- Month-to-month: **42.7%**
- One year: **11.3%**
- Two year: **2.8%**
- Explain other high-churn service/payment segments from the statistical tables.

## Slide 7 — Statistical Evidence
- Chi-square tests for categorical associations.
- Welch's t-tests for numeric group differences.
- Benjamini–Hochberg correction for multiple categorical tests.
- Say **associated with churn**, not **causes churn**.

## Slide 8 — Churn Prediction
- Logistic Regression with preprocessing pipeline.
- Holdout ROC-AUC: **0.845**
- Recall: **0.805**
- Use risk probabilities for prioritization, not automatic customer actions.

## Slide 9 — Retention Strategy
1. Early onboarding/activation.
2. High-risk/high-value customer outreach.
3. Contract conversion experiments.
4. Support/service interventions.
5. Cancellation-reason capture.

## Slide 10 — Conclusion & Next Steps
- Operationalize risk scoring on active customers.
- Add signup date, region, usage, engagement, support and cancellation-reason data.
- Build a true calendar cohort model and CLV model.
- Measure interventions using controlled experiments.
