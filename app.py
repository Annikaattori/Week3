import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.datasets import load_diabetes


st.set_page_config(page_title="Week 3 Correlation Explorer", layout="wide")

st.title("Week 3: Correlation Analysis Explorer")

st.markdown(
    """
This app explores correlations in the **Diabetes** dataset (442 observations, 10 numeric predictors,
and a numeric target). It includes EDA, normality checks, correlation analysis, and visualizations.
"""
)

dataset = load_diabetes(as_frame=True)
data = dataset.frame.rename(columns={"target": "disease_progression"})

numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

st.header("Dataset Preview")
st.dataframe(data.head(10))

st.subheader("Missing Values")
missing = data.isna().sum()
st.write(missing)

st.subheader("Descriptive Statistics")
st.dataframe(data.describe().T)

st.header("Exploratory Visualizations")

default_hist_columns = numeric_columns[:4]
hist_columns = st.multiselect(
    "Select variables for histograms", numeric_columns, default=default_hist_columns
)

if hist_columns:
    hist_fig, hist_axes = plt.subplots(
        nrows=len(hist_columns), ncols=1, figsize=(8, 3 * len(hist_columns))
    )
    if len(hist_columns) == 1:
        hist_axes = [hist_axes]
    for ax, column in zip(hist_axes, hist_columns):
        sns.histplot(data[column], kde=True, ax=ax, color="#4C78A8")
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Count")
    st.pyplot(hist_fig)

scatter_x = st.selectbox("Scatter plot X-axis", numeric_columns, index=0)
scatter_y = st.selectbox("Scatter plot Y-axis", numeric_columns, index=1)

scatter_fig, scatter_ax = plt.subplots(figsize=(6, 4))
sns.regplot(
    data=data,
    x=scatter_x,
    y=scatter_y,
    scatter_kws={"alpha": 0.6},
    line_kws={"color": "#F58518"},
    ax=scatter_ax,
)
scatter_ax.set_title(f"Scatter Plot with Trend Line: {scatter_x} vs {scatter_y}")
scatter_ax.set_xlabel(scatter_x)
scatter_ax.set_ylabel(scatter_y)
st.pyplot(scatter_fig)

st.header("Normality Checks")

selected_for_normality = st.multiselect(
    "Variables to test for normality (Shapiro-Wilk)",
    numeric_columns,
    default=numeric_columns[:5],
)

normality_results = []
for column in selected_for_normality:
    stat, p_value = stats.shapiro(data[column])
    normality_results.append(
        {
            "variable": column,
            "shapiro_stat": stat,
            "p_value": p_value,
            "normal_at_0.05": p_value > 0.05,
        }
    )

normality_df = pd.DataFrame(normality_results)
st.dataframe(normality_df)

st.header("Correlation Analysis")

correlation_method = st.selectbox("Correlation method", ["Pearson", "Spearman"])

correlation_matrix = data[numeric_columns].corr(method=correlation_method.lower())

st.subheader("Correlation Matrix")
st.dataframe(correlation_matrix.style.format("{:.2f}"))

heatmap_fig, heatmap_ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=heatmap_ax,
)
heatmap_ax.set_title(f"{correlation_method} Correlation Heatmap")
st.pyplot(heatmap_fig)

st.subheader("Key Correlations")

upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
stacked = upper_triangle.stack().sort_values()

strong_negative = stacked.iloc[0]
strong_positive = stacked.iloc[-1]

neg_pair = strong_negative.name
pos_pair = strong_positive.name

st.markdown(
    f"""
**Strong Negative Correlation:** {neg_pair[0]} vs {neg_pair[1]} = {strong_negative:.2f}

**Strong Positive Correlation:** {pos_pair[0]} vs {pos_pair[1]} = {strong_positive:.2f}
"""
)

st.header("Interpretation & Discussion")

st.markdown(
    f"""
- The strongest positive correlation is between **{pos_pair[0]}** and **{pos_pair[1]}**
  (r = {strong_positive:.2f}), suggesting they tend to increase together.
- The strongest negative correlation is between **{neg_pair[0]}** and **{neg_pair[1]}**
  (r = {strong_negative:.2f}), indicating one tends to decrease as the other increases.
- Several variables show weaker correlations, which may indicate limited linear relationships
  or influence from confounding variables (e.g., age or BMI interacting with other predictors).
- Because the dataset is standardized, correlations reflect relative directional relationships
  rather than raw-unit effects.
- If normality tests indicate non-normal variables, Spearman correlations can be more robust to
  non-linear or non-normal patterns.
"""
)
