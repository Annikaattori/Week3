from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from scipy import stats


st.set_page_config(page_title="Week 3 Correlation Explorer", layout="wide")

st.title("Week 3: Correlation Analysis Explorer")

st.markdown(
    """
This app explores correlations in the **Retail Sales Dataset** from Kaggle. It includes EDA,
normality checks, correlation analysis, and visualizations.
"""
)

@st.cache_data(show_spinner=False)
def load_retail_sales_data() -> pd.DataFrame:
    dataset_path = kagglehub.dataset_download("terencekatua/retail-sales-dataset")
    data_path = Path(dataset_path)
    csv_files = list(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the downloaded dataset directory.")
    return pd.read_csv(csv_files[0])


data = load_retail_sales_data()

numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_columns:
    st.warning("No numeric columns were detected in the dataset.")

st.caption(f"Loaded dataset with {data.shape[0]:,} rows and {data.shape[1]:,} columns.")

st.header("Dataset Preview")
st.dataframe(data.head(10))

st.subheader("Missing & Empty Values")

# Count NaN values per column
na_counts = data.isna().sum()
st.write("NaN counts per column:")
st.write(na_counts)

# Count zeros in numeric columns
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    zero_counts = (data[numeric_cols] == 0).sum()
    st.write("Zero counts per numeric column:")
    st.write(zero_counts)
else:
    zero_counts = pd.Series(dtype=int)
    st.write("No numeric columns detected for zero-value checks.")

# Count empty or whitespace-only strings in object (text) columns
obj_cols = data.select_dtypes(include=[object]).columns.tolist()
blank_counts = {}
if obj_cols:
    for col in obj_cols:
        blank = data[col].apply(lambda x: isinstance(x, str) and x.strip() == "").sum()
        blank_counts[col] = int(blank)
    st.write("Empty/whitespace-only string counts per text column:")
    st.write(pd.Series(blank_counts))
else:
    st.write("No text columns to check for empty strings.")

# Totals summary
total_nans = int(na_counts.sum())
total_zeros = int(zero_counts.sum()) if not zero_counts.empty else 0
total_blanks = int(sum(blank_counts.values())) if blank_counts else 0
st.markdown(
    f"**Totals — NaNs:** {total_nans:,} — **Zeros:** {total_zeros:,} — **Empty strings:** {total_blanks:,}"
)

st.subheader("Descriptive Statistics")
st.dataframe(data.describe().T)

st.header("Exploratory Visualizations")

if numeric_columns:
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

if numeric_columns:
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

if numeric_columns:
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

    if stacked.empty:
        st.info("Not enough numeric columns to compute key pairwise correlations.")
        neg_pair = pos_pair = (None, None)
        strong_negative = strong_positive = np.nan
    else:
        strong_negative = stacked.iloc[0]
        strong_positive = stacked.iloc[-1]
        neg_pair = stacked.index[0]
        pos_pair = stacked.index[-1]

        st.markdown(
            f"""
**Strong Negative Correlation:** {neg_pair[0]} vs {neg_pair[1]} = {strong_negative:.2f}

**Strong Positive Correlation:** {pos_pair[0]} vs {pos_pair[1]} = {strong_positive:.2f}
"""
        )

st.header("Interpretation & Discussion")

if numeric_columns and not stacked.empty:
        st.markdown(
                f"""
- The strongest positive correlation is between **{pos_pair[0]}** and **{pos_pair[1]}**
    (r = {strong_positive:.2f}), suggesting they tend to increase together.
- The strongest negative correlation is between **{neg_pair[0]}** and **{neg_pair[1]}**
    (r = {strong_negative:.2f}), indicating one tends to decrease as the other increases.
- Several variables show weaker correlations, which may indicate limited linear relationships
    or influence from confounding variables.
- Because normality tests in this dataset indicate non-normal variables, Spearman correlations can be more robust to
    non-linear or non-normal patterns.
"""
        )
else:
        st.markdown(
                """
- Not enough numeric columns to provide interpretation of pairwise correlations.
"""
        )
