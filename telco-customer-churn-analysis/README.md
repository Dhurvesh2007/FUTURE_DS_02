# 📊 Telco Customer Churn Analysis & Prediction

> End-to-end customer churn analytics project using Python, statistical analysis, customer segmentation, and machine learning to identify churn patterns, quantify recurring-revenue exposure, and recommend retention strategies.

## 🎯 Business Objective

Customer churn impacts recurring revenue and customer lifetime value. This project answers:

- What is the overall churn and retention rate?
- Which customer lifecycle stages have the highest churn?
- Which subscription/contract segments are most at risk?
- Which services and payment methods are associated with churn?
- How does customer tenure relate to churn?
- How much recurring monthly revenue is associated with churned customers?
- Which variables are statistically associated with churn?
- Can customers be assigned a practical churn-risk score?
- What retention actions should a subscription business prioritize?

## 📌 Key Results

| KPI | Result |
|---|---:|
| Total customers | **7,043** |
| Churned customers | **1,869** |
| Retained customers | **5,174** |
| Overall churn rate | **26.5%** |
| Overall retention rate | **73.5%** |
| Average tenure | **32.4 months** |
| Average monthly charges | **$64.76** |
| Monthly charges associated with churned customers | **$139,130.85** |

### Lifecycle pattern

| Tenure | Churn Rate |
|---|---:|
| 0–6 months | **52.9%** |
| 7–12 months | **35.9%** |
| 13–24 months | **28.7%** |
| 25–48 months | **20.4%** |
| 49–72 months | **9.5%** |

### Contract pattern

| Contract | Churn Rate |
|---|---:|
| Month-to-month | **42.7%** |
| One year | **11.3%** |
| Two year | **2.8%** |

## 🧠 Workflow

```text
Raw Dataset → Data Quality → Cleaning → EDA → Churn/Retention
→ Lifecycle Cohorts → Segment Analysis → Lifetime/Revenue
→ Statistical Testing → Logistic Regression → Risk Scoring
→ Business Recommendations
```

## 🧹 Data Cleaning

- Missing-value audit
- Duplicate customer-ID check
- Text normalization
- `TotalCharges` conversion to numeric
- Missing `TotalCharges` handling
- `ChurnFlag` creation
- Tenure-band creation
- Revenue/value feature engineering

## 📈 Analysis

- Overall churn and retention
- Churn by tenure/lifecycle
- Churn by contract
- Churn by internet service, payment method, billing and service attributes
- Customer lifetime/value patterns
- Monthly recurring-revenue exposure from churn
- Chi-square tests
- T-tests
- Logistic-regression churn model
- Odds-ratio analysis
- Customer-level churn-risk scoring

## 💡 Business Recommendations

1. Strengthen early customer onboarding and activation.
2. Prioritize high-risk, high-value customers for retention outreach.
3. Test suitable month-to-month to longer-contract conversion strategies.
4. Investigate service/support gaps in high-churn segments.
5. Capture explicit cancellation reasons and exit-survey feedback.
6. Add product usage, engagement, support, payment and acquisition data.
7. Validate retention actions through controlled experiments rather than assuming association implies causation.

## ⚠️ Data Limitations

The supplied dataset does **not** contain signup date, region, or explicit churn reason. Therefore, exact calendar-month cohorts, regional churn analysis, and stated churn reasons cannot be calculated from this file. Tenure bands are used as lifecycle cohorts instead of fabricating unavailable information.

## 🗂️ Repository Structure

```text
telco-customer-churn-analysis/
├── data/
├── notebooks/
├── src/
├── outputs/
├── README.md
├── requirements.txt
├── project_report.md
├── presentation_outline.md
├── GITHUB_UPLOAD_GUIDE.md
├── LICENSE
└── .gitignore
```

## 🛠️ Technologies

Python · Pandas · NumPy · Matplotlib · Seaborn · SciPy · Scikit-learn · Jupyter Notebook · Logistic Regression · Statistical Testing

## ▶️ Run Locally

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/Telco_Customer_Churn_End_to_End_Project.ipynb` and run all cells.

## ⭐ Future Improvements

- XGBoost / LightGBM
- SHAP explainability
- Survival analysis
- True calendar-month cohort retention
- Customer Lifetime Value modeling
- Churn-reason classification
- Power BI dashboard
- API deployment
- Automated scoring
- Retention A/B testing

## 📌 Portfolio Note

This is an educational/internship analytics project. Model outputs should be validated against current business data before operational use.
