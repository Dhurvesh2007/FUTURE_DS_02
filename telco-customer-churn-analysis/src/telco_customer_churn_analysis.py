"""Telco Customer Churn — end-to-end internship analysis.

The script is path-safe: it resolves the project root from this file, so it can
be executed from any working directory after cloning the repository.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
TABLE_DIR = OUTPUT_DIR / "tables"
OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def segment_churn(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Return customer count and churn rate for a categorical segment."""
    return (
        data.groupby(col, dropna=False)["ChurnFlag"]
        .agg(Customers="count", ChurnRate="mean")
        .reset_index()
        .sort_values("ChurnRate", ascending=False)
    )


def save_bar(series: pd.Series, title: str, ylabel: str, xlabel: str, path: Path,
             figsize=(8, 5), rotation=0, percent=False) -> None:
    plt.figure(figsize=figsize)
    ax = series.plot(kind="bar")
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="x", rotation=rotation)
    if percent:
        ax.set_yticklabels([f"{v:.0%}" for v in ax.get_yticks()])
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# 1. Load and audit
# -----------------------------------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH)
print(f"Rows: {df_raw.shape[0]:,}")
print(f"Columns: {df_raw.shape[1]:,}")

quality_before = pd.DataFrame({
    "dtype": df_raw.dtypes.astype(str),
    "missing_count": df_raw.isna().sum(),
    "missing_pct": (df_raw.isna().mean() * 100).round(2),
    "unique_values": df_raw.nunique()
})

print(f"Duplicate rows: {df_raw.duplicated().sum():,}")
print(f"Unique customer IDs: {df_raw['customerID'].nunique():,}")

# -----------------------------------------------------------------------------
# 2. Clean and feature engineer
# -----------------------------------------------------------------------------
df = df_raw.copy()
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
duplicates_removed = int(df.duplicated(subset="customerID").sum())
df = df.drop_duplicates(subset="customerID", keep="first").copy()
missing_totalcharges_before = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

tenure_labels = ["0–6 months", "7–12 months", "13–24 months", "25–48 months", "49–72 months"]
df["TenureBand"] = pd.cut(
    df["tenure"], bins=[-1, 6, 12, 24, 48, 72], labels=tenure_labels
)
df["AnnualizedMonthlyRevenue"] = df["MonthlyCharges"] * 12
df["AvgMonthlyValue"] = np.where(
    df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
)

quality_after = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_count": df.isna().sum(),
    "missing_pct": (df.isna().mean() * 100).round(2)
})

# -----------------------------------------------------------------------------
# 3. Executive KPIs
# -----------------------------------------------------------------------------
total_customers = len(df)
churned = int(df["ChurnFlag"].sum())
retained = total_customers - churned
churn_rate = churned / total_customers
retention_rate = retained / total_customers

kpis = pd.DataFrame({
    "KPI": [
        "Customers", "Churned Customers", "Retained Customers",
        "Churn Rate", "Retention Rate", "Average Tenure (months)",
        "Average Monthly Charges", "Average Total Charges"
    ],
    "Value": [
        total_customers, churned, retained, churn_rate, retention_rate,
        df["tenure"].mean(), df["MonthlyCharges"].mean(), df["TotalCharges"].mean()
    ]
})
kpis.to_csv(TABLE_DIR / "executive_kpis.csv", index=False)

# -----------------------------------------------------------------------------
# 4. Churn and lifecycle analysis
# -----------------------------------------------------------------------------
overall = df["Churn"].value_counts(normalize=True).reindex(["No", "Yes"])
save_bar(
    overall, "Overall Churn vs Retention", "Customer Share", "Outcome",
    CHART_DIR / "01_overall_churn_retention.png", rotation=0, percent=True
)

tenure_churn = (
    df.groupby("TenureBand", observed=False)["ChurnFlag"]
    .agg(Customers="count", ChurnRate="mean")
    .reset_index()
)
tenure_churn.to_csv(TABLE_DIR / "tenure_lifecycle_churn.csv", index=False)
save_bar(
    tenure_churn.set_index("TenureBand")["ChurnRate"],
    "Churn Rate by Customer Tenure Band", "Churn Rate", "Tenure",
    CHART_DIR / "02_churn_by_tenure.png", figsize=(9, 5), rotation=20, percent=True
)

contract_churn = segment_churn(df, "Contract")
contract_churn.to_csv(TABLE_DIR / "contract_churn.csv", index=False)
save_bar(
    contract_churn.set_index("Contract")["ChurnRate"],
    "Churn Rate by Contract Type", "Churn Rate", "Contract",
    CHART_DIR / "03_churn_by_contract.png", rotation=0, percent=True
)

lifetime_summary = df.groupby("TenureBand", observed=False).agg(
    Customers=("customerID", "count"),
    AvgTenure=("tenure", "mean"),
    AvgMonthlyCharges=("MonthlyCharges", "mean"),
    AvgTotalCharges=("TotalCharges", "mean"),
    ChurnRate=("ChurnFlag", "mean")
).reset_index()
lifetime_summary.to_csv(TABLE_DIR / "lifecycle_value_summary.csv", index=False)

for metric, filename, title, ylabel in [
    ("tenure", "04_tenure_churn_boxplot.png", "Customer Tenure: Churned vs Retained", "Tenure (months)"),
    ("TotalCharges", "05_totalcharges_churn_boxplot.png", "Total Charges: Churned vs Retained", "Total Charges")
]:
    plt.figure(figsize=(9, 5))
    df.boxplot(column=metric, by="Churn")
    plt.title(title, fontweight="bold")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close()

# Revenue exposure: distinguish historical churned revenue from active high-risk revenue.
monthly_charges_churned = df.loc[df["ChurnFlag"] == 1, "MonthlyCharges"].sum()
total_monthly_charges = df["MonthlyCharges"].sum()
revenue_summary = pd.DataFrame({
    "Metric": [
        "Total Monthly Charges",
        "Monthly Charges Associated With Churned Customers",
        "Share of Monthly Charges Associated With Churned Customers"
    ],
    "Value": [total_monthly_charges, monthly_charges_churned,
              monthly_charges_churned / total_monthly_charges]
})
revenue_summary.to_csv(TABLE_DIR / "revenue_exposure_summary.csv", index=False)

# -----------------------------------------------------------------------------
# 5. Statistical testing with multiple-testing correction
# -----------------------------------------------------------------------------
categorical_features_for_test = [
    "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
    "SeniorCitizen", "Partner", "Dependents", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines"
]

chi_results = []
for col in categorical_features_for_test:
    table = pd.crosstab(df[col], df["Churn"])
    chi2, p, dof, expected = chi2_contingency(table)
    chi_results.append({"Feature": col, "Chi2": chi2, "p_value": p,
                        "Significant_at_5pct": p < 0.05})
chi_results = pd.DataFrame(chi_results).sort_values("p_value").reset_index(drop=True)
# Benjamini-Hochberg false-discovery-rate correction.
m = len(chi_results)
chi_results["BH_adjusted_p"] = np.minimum.accumulate(
    (chi_results["p_value"] * m / (np.arange(m) + 1))[::-1]
)[::-1]
chi_results["Significant_after_BH"] = chi_results["BH_adjusted_p"] < 0.05
chi_results.to_csv(TABLE_DIR / "chi_square_results.csv", index=False)

ttest_results = []
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    churned_values = df.loc[df["ChurnFlag"] == 1, col]
    retained_values = df.loc[df["ChurnFlag"] == 0, col]
    stat, p = ttest_ind(churned_values, retained_values, equal_var=False)
    ttest_results.append({
        "Feature": col,
        "ChurnedMean": churned_values.mean(),
        "RetainedMean": retained_values.mean(),
        "t_statistic": stat,
        "p_value": p,
        "Significant_at_5pct": p < 0.05
    })
ttest_results = pd.DataFrame(ttest_results)
ttest_results.to_csv(TABLE_DIR / "ttest_results.csv", index=False)

# -----------------------------------------------------------------------------
# 6. Logistic regression: evaluate out-of-sample, then refit for final scoring
# -----------------------------------------------------------------------------
# Exclude historical/redundant value aggregates from the predictive feature set.
# This keeps the intervention model closer to information available at scoring time.
model_drop = [
    "customerID", "Churn", "ChurnFlag",
    "TotalCharges", "AnnualizedMonthlyRevenue", "AvgMonthlyValue"
]
X = df.drop(columns=model_drop)
y = df["ChurnFlag"]

categorical_features = X.select_dtypes(include="object").columns.tolist()
if "TenureBand" in X.columns:
    categorical_features.append("TenureBand")
categorical_features = list(dict.fromkeys(categorical_features))
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ]), categorical_features)
])

model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]

metrics = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
    "Value": [
        accuracy_score(y_test, pred), precision_score(y_test, pred),
        recall_score(y_test, pred), f1_score(y_test, pred),
        roc_auc_score(y_test, proba)
    ]
})
metrics.to_csv(TABLE_DIR / "model_metrics.csv", index=False)

cm = pd.DataFrame(
    confusion_matrix(y_test, pred),
    index=["Actual No Churn", "Actual Churn"],
    columns=["Pred No Churn", "Pred Churn"]
)
cm.to_csv(TABLE_DIR / "confusion_matrix.csv")

with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, pred))

# Driver ranking is descriptive association, not causal inference.
feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
coefficients = pipe.named_steps["model"].coef_[0]
importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "OddsRatio": np.exp(coefficients)
}).sort_values("Coefficient", ascending=False)
importance.to_csv(TABLE_DIR / "logistic_driver_ranking.csv", index=False)

top_plot = pd.concat([importance.head(8), importance.tail(8)]).sort_values("Coefficient")
plt.figure(figsize=(10, 7))
top_plot.set_index("Feature")["Coefficient"].plot(kind="barh")
plt.title("Logistic Regression Churn Associations", fontweight="bold")
plt.xlabel("Coefficient")
plt.ylabel("")
plt.tight_layout()
plt.savefig(CHART_DIR / "06_logistic_churn_drivers.png", dpi=220, bbox_inches="tight")
plt.close()

# Refit on all available historical records before generating the final portfolio score.
final_pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
final_pipe.fit(X, y)
all_scores = final_pipe.predict_proba(X)[:, 1]

df_scored = df.copy()
df_scored["ChurnRiskScore"] = all_scores
df_scored["RiskTier"] = pd.cut(
    df_scored["ChurnRiskScore"],
    bins=[-0.01, 0.30, 0.60, 1.00],
    labels=["Low", "Medium", "High"]
)

risk_summary = (
    df_scored["RiskTier"].value_counts(sort=False)
    .rename_axis("RiskTier")
    .reset_index(name="Customers")
)
risk_summary["Share"] = risk_summary["Customers"] / len(df_scored)
risk_summary.to_csv(TABLE_DIR / "risk_tier_summary.csv", index=False)

retention_priority = df_scored.sort_values("ChurnRiskScore", ascending=False)[[
    "customerID", "Contract", "tenure", "MonthlyCharges", "TotalCharges",
    "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity",
    "Churn", "ChurnRiskScore", "RiskTier"
]]
retention_priority.to_csv(TABLE_DIR / "customer_churn_risk_scores.csv", index=False)

# -----------------------------------------------------------------------------
# 7. Validation summary and project metadata
# -----------------------------------------------------------------------------
validation = pd.DataFrame({
    "Check": [
        "Original rows", "Cleaned rows", "Duplicate customer IDs after cleaning",
        "Missing target values", "Blank TotalCharges handled", "Dataset columns",
        "Model evaluation split", "Random state"
    ],
    "Result": [
        len(df_raw), len(df), int(df["customerID"].duplicated().sum()),
        int(df["ChurnFlag"].isna().sum()), missing_totalcharges_before,
        df_raw.shape[1], "80% train / 20% test", RANDOM_STATE
    ]
})
validation.to_csv(TABLE_DIR / "data_quality_validation.csv", index=False)

print("\n=== PROJECT SUMMARY ===")
print(f"Customers: {total_customers:,}")
print(f"Churned: {churned:,} ({churn_rate:.1%})")
print(f"Retained: {retained:,} ({retention_rate:.1%})")
print(f"Average tenure: {df['tenure'].mean():.1f} months")
print(f"Average monthly charges: ${df['MonthlyCharges'].mean():,.2f}")
print(f"Historical monthly charges associated with churned customers: ${monthly_charges_churned:,.2f}")
print(f"Model ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
print(f"Model recall: {recall_score(y_test, pred):.3f}")
print(f"Outputs: {OUTPUT_DIR}")
