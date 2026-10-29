"""
Week 2: Exploratory Data Analysis and Visualization
Dataset: scikit-learn diabetes dataset (442 observations, 10 standardized features)
Run: pip install -r requirements.txt
     python eda_analysis.py
Outputs: diabetes_eda_dataset.csv and visualizations/*.png
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

OUT = Path("eda_outputs")
PLOTS = OUT / "visualizations"
PLOTS.mkdir(parents=True, exist_ok=True)

# 1. Load a documented public dataset available through scikit-learn.
# The target is a quantitative measure of disease progression one year after baseline.
data = load_diabetes(as_frame=True)
df = data.frame.rename(columns={"target": "disease_progression"}).copy()

# 2. Initial inspection
print("Shape:", df.shape)
print("\\nFirst five rows:\\n", df.head())
print("\\nData types:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum())
print("\\nDescriptive statistics:\\n", df.describe().T)

# 3. Duplicate and missing-value checks
print("Duplicate rows:", df.duplicated().sum())
# No imputation is performed unless missing values are found and justified.
# The original dataset is complete, so the analysis retains all rows.

# 4. Correlation analysis
corr = df.corr(numeric_only=True)
print("\\nCorrelations with target:\\n",
      corr["disease_progression"].drop("disease_progression")
      .sort_values(key=abs, ascending=False))

# 5. Visualization: target distribution
plt.figure(figsize=(8, 5))
plt.hist(df["disease_progression"], bins=25, edgecolor="black")
plt.title("Distribution of Disease Progression Target")
plt.xlabel("Disease progression measure")
plt.ylabel("Number of observations")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(PLOTS / "01_target_distribution.png", dpi=180)
plt.close()

# 6. Correlation heatmap using Matplotlib
plt.figure(figsize=(10, 8))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=75, ha="right", fontsize=8)
plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)
plt.title("Correlation Matrix of Diabetes Dataset Features")
plt.colorbar(im, label="Pearson correlation")
plt.tight_layout()
plt.savefig(PLOTS / "02_correlation_heatmap.png", dpi=180)
plt.close()

# 7. Scatter plot for the feature most correlated with the target
target_corr = corr["disease_progression"].drop("disease_progression")
feature = target_corr.abs().idxmax()
plt.figure(figsize=(7, 5))
plt.scatter(df[feature], df["disease_progression"], alpha=0.55)
m, b = __import__("numpy").polyfit(df[feature], df["disease_progression"], 1)
xs = __import__("numpy").linspace(df[feature].min(), df[feature].max(), 100)
plt.plot(xs, m * xs + b, linewidth=2, label="Least-squares trend")
plt.title(f"{feature} vs Disease Progression")
plt.xlabel(f"{feature} (standardized feature)")
plt.ylabel("Disease progression measure")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PLOTS / "03_top_feature_scatter.png", dpi=180)
plt.close()

# 8. Compare selected feature distributions
selected = ["bmi", "bp", "s5", "s6"]
plt.figure(figsize=(9, 5))
for col in selected:
    plt.hist(df[col], bins=25, alpha=0.45, label=col)
plt.title("Distributions of Selected Standardized Features")
plt.xlabel("Standardized feature value")
plt.ylabel("Frequency")
plt.legend(title="Feature")
plt.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.savefig(PLOTS / "04_feature_distributions.png", dpi=180)
plt.close()

# 9. Boxplots for potential outlier screening
plt.figure(figsize=(8, 5))
plt.boxplot([df[c] for c in selected],
            tick_labels=selected,
            showfliers=True)
plt.title("Boxplots of Selected Features")
plt.xlabel("Feature")
plt.ylabel("Standardized feature value")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(PLOTS / "05_feature_boxplots.png", dpi=180)
plt.close()

# 10. Aggregation: target mean by BMI quartile
df["bmi_group"] = pd.qcut(
    df["bmi"], 4, labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]
)
summary = df.groupby("bmi_group", observed=False)["disease_progression"].agg(
    ["count", "mean", "median", "std"]
)
print("\\nDisease progression by BMI quartile:\\n", summary)

plt.figure(figsize=(7, 5))
plt.bar(summary.index.astype(str), summary["mean"])
plt.title("Mean Disease Progression by BMI Quartile")
plt.xlabel("BMI quartile (standardized BMI)")
plt.ylabel("Mean disease progression")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(PLOTS / "06_target_by_bmi_quartile.png", dpi=180)
plt.close()

# Export analysis tables
df.to_csv(OUT / "diabetes_eda_dataset_with_bmi_group.csv", index=False)
desc = df.drop(columns=["bmi_group"]).describe().T
desc.to_csv(OUT / "descriptive_statistics.csv")
corr.to_csv(OUT / "correlation_matrix.csv")
summary.to_csv(OUT / "bmi_quartile_summary.csv")
print(f"\\nAnalysis complete. Outputs saved in: {OUT.resolve()}")
