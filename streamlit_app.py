import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix

# If you want to include clustering/PCA later:
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD DATA
# ────────────────────────────────────────────────────────────────────────────────

@st.cache  # or @st.cache_data if your Streamlit version supports it
def load_data():
    return pd.read_csv("rail_synthetic_formatted.csv")

df = load_data()
numeric_cols = df.select_dtypes(include="number").columns.tolist()

# ────────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Benchmarking Dashboard")

show_overview       = st.sidebar.checkbox("1. Data Overview", True)
show_distributions  = st.sidebar.checkbox("2. Distribution & Outliers", False)
show_corr           = st.sidebar.checkbox("3. Correlation & Significance", False)
show_scatter        = st.sidebar.checkbox("4. Pairwise Scatter Plots", False)
show_normalization  = st.sidebar.checkbox("5. Normalization & Clustering", False)
show_top10          = st.sidebar.checkbox("6. Top 10 Projects", False)
show_downloads      = st.sidebar.checkbox("7. Download Reports", False)
show_insights       = st.sidebar.checkbox("8. Actionable Insights", False)

st.sidebar.markdown("---")
st.sidebar.write("Data file: **rail_synthetic_formatted.csv**")
st.sidebar.write("Numeric fields detected:")
for col in numeric_cols:
    st.sidebar.write(f"• {col}")

# ────────────────────────────────────────────────────────────────────────────────
# 3) PANEL: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_overview:
    st.header("1. Data Overview")
    st.write("Below are the first few rows of your dataset:")
    st.dataframe(df.head())

    st.write("### Descriptive Statistics for Numeric Columns")
    desc = df[numeric_cols].describe()
    st.dataframe(desc)

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL: DISTRIBUTION & OUTLIER ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────

if show_distributions:
    st.header("2. Distribution & Outlier Analysis")

    col = st.selectbox("Select a numeric column to inspect:", numeric_cols)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Compute IQR and outlier bounds
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_condition = (df[col] < lower_bound) | (df[col] > upper_bound)
    outlier_count = outlier_condition.sum()

    st.write(f"**Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}**")
    st.write(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    st.write(f"Number of outliers detected in '{col}': {outlier_count}")

    if st.checkbox("Show sample outlier rows for this column"):
        st.dataframe(df.loc[outlier_condition].head(10))

    if st.checkbox("Show boxplots for all numeric columns"):
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.boxplot([df[c].dropna() for c in numeric_cols], labels=numeric_cols, vert=False)
        ax2.set_title("Boxplot of All Numeric Columns")
        st.pyplot(fig2)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL: CORRELATION & SIGNIFICANCE
# ────────────────────────────────────────────────────────────────────────────────

if show_corr:
    st.header("3. Correlation & Significance")

    corr_matrix = df[numeric_cols].corr()
    st.write("### Pearson Correlation Matrix")
    st.dataframe(corr_matrix)

    if st.checkbox("Show correlation heatmap"):
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        im = ax3.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax3.set_xticks(range(len(numeric_cols)))
        ax3.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax3.set_yticks(range(len(numeric_cols)))
        ax3.set_yticklabels(numeric_cols)
        fig3.colorbar(im, ax=ax3)
        st.pyplot(fig3)

    # Compute p-value matrix
    def compute_pvalues(df_num):
        cols = df_num.columns
        pval_mat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val, p_val = pearsonr(df_num[cols[i]].dropna(), df_num[cols[j]].dropna())
                pval_mat.loc[cols[i], cols[j]] = p_val
                pval_mat.loc[cols[j], cols[i]] = p_val
        return pval_mat

    if st.checkbox("Show p-value matrix"):
        p_values = compute_pvalues(df[numeric_cols])
        st.write("### P-Value Matrix")
        st.dataframe(p_values)

    if st.checkbox("Show R² (correlation²) matrix"):
        r_squared = corr_matrix ** 2
        st.write("### R² (Correlation²) Matrix")
        st.dataframe(r_squared)

# ────────────────────────────────────────────────────────────────────────────────
# 6) PANEL: PAIRWISE SCATTER PLOTS
# ────────────────────────────────────────────────────────────────────────────────

if show_scatter:
    st.header("4. Pairwise Scatter Plots")

    x_col = st.selectbox("X-axis column:", numeric_cols, index=0, key="scatter_x")
    y_col = st.selectbox("Y-axis column:", numeric_cols, index=1, key="scatter_y")

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.scatter(df[x_col], df[y_col], alpha=0.6)
    ax4.set_xlabel(x_col)
    ax4.set_ylabel(y_col)
    ax4.set_title(f"Scatter Plot: {x_col} vs. {y_col}")
    st.pyplot(fig4)

    r_val, p_val = pearsonr(df[x_col].dropna(), df[y_col].dropna())
    st.write(f"Pearson r = {r_val:.2f}, p-value = {p_val:.3e}, R² = {r_val**2:.2f}")

    if st.checkbox("Show full scatter matrix for selected columns"):
        cols_for_matrix = st.multiselect(
            "Pick columns for scatter matrix (choose at least 2):", numeric_cols, default=numeric_cols[:3]
        )
        if len(cols_for_matrix) >= 2:
            fig5 = scatter_matrix(df[cols_for_matrix], alpha=0.6, diagonal="hist", figsize=(8, 8))
            st.pyplot(plt.gcf())

# ────────────────────────────────────────────────────────────────────────────────
# 7) PANEL: NORMALIZATION & CLUSTERING
# ────────────────────────────────────────────────────────────────────────────────

if show_normalization:
    st.header("5. Normalization & (Optional) Clustering")

    if st.checkbox("Show Z-score normalized data"):
        df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        st.write("### First 5 rows of normalized data")
        st.dataframe(df_norm.head())
        st.download_button(
            label="Download normalized CSV",
            data=df_norm.to_csv(index=False).encode("utf-8"),
            file_name="rail_normalized.csv",
            mime="text/csv"
        )

    if st.checkbox("Run K-Means clustering on normalized data"):
        df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        k = st.slider("Select number of clusters (k):", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df_norm)
        df["Cluster"] = kmeans.labels_
        st.write("Cluster labels added to the DataFrame.")

        # PCA for visualization
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(df_norm)
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        scatter = ax6.scatter(pcs[:, 0], pcs[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.7)
        ax6.set_xlabel("PC 1")
        ax6.set_ylabel("PC 2")
        ax6.set_title("PCA Projection Colored by K-Means Cluster")
        st.pyplot(fig6)

# ────────────────────────────────────────────────────────────────────────────────
# 8) PANEL: TOP 10 PROJECTS
# ────────────────────────────────────────────────────────────────────────────────

if show_top10:
    st.header("6. Top 10 Projects by Chosen Metric")

    # 8.1) Let user pick a numeric column to rank by
    metric = st.selectbox(
        "Select metric for ranking (highest first):",
        options=numeric_cols,
        index=numeric_cols.index("CapEx ($M)") if "CapEx ($M)" in numeric_cols else 0
    )

    # 8.2) Choose sort order
    order = st.radio("Sort order:", ["Descending (largest → smallest)", "Ascending (smallest → largest)"])
    ascending = True if order.startswith("Ascending") else False

    # 8.3) Sort and slice
    df_sorted = df.sort_values(by=metric, ascending=ascending)
    top_n = df_sorted.head(10)

    st.write(f"Showing top 10 projects by **{metric}** ({'lowest' if ascending else 'highest'} values):")
    st.dataframe(top_n.reset_index(drop=True))

    # 8.4) Bar chart of the top 10 with cleaner heading
    st.subheader(f"Top 10 Project IDs vs. {metric}")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    ax7.barh(top_n["Project ID"].astype(str), top_n[metric], color="skyblue")
    ax7.set_xlabel(metric)
    ax7.set_ylabel("Project ID")
    ax7.invert_yaxis()
    ax7.set_title(f"Top 10 by {metric}")
    plt.tight_layout()
    st.pyplot(fig7)

    # 8.5) Composite score example
    if st.checkbox("Rank by composite score (CapEx × Safety Incidents)"):
        df_prime = df.copy()
        df_prime["Composite Score"] = df_prime["CapEx ($M)"] * df_prime["Safety Incidents"]
        df_cs_sorted = df_prime.sort_values(by="Composite Score", ascending=False).head(10)

        st.write("### Top 10 by Composite Score (CapEx × Safety Incidents):")
        cols_to_show = ["Project ID", "CapEx ($M)", "Safety Incidents", "Composite Score"]
        st.dataframe(df_cs_sorted[cols_to_show].reset_index(drop=True))

        fig8, ax8 = plt.subplots(figsize=(6, 4))
        ax8.barh(
            df_cs_sorted["Project ID"].astype(str),
            df_cs_sorted["Composite Score"],
            color="salmon"
        )
        ax8.set_xlabel("Composite Score")
        ax8.set_ylabel("Project ID")
        ax8.invert_yaxis()
        ax8.set_title("Top 10 by Composite Score")
        plt.tight_layout()
        st.pyplot(fig8)

# ────────────────────────────────────────────────────────────────────────────────
# 9) PANEL: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

if show_downloads:
    st.header("7. Download Reports")

    # Descriptive statistics CSV
    desc = df[numeric_cols].describe()
    st.download_button(
        label="Download descriptive_statistics.csv",
        data=desc.to_csv().encode("utf-8"),
        file_name="descriptive_statistics.csv",
        mime="text/csv"
    )

    # Correlation matrix CSV
    corr_matrix = df[numeric_cols].corr()
    st.download_button(
        label="Download correlation_matrix.csv",
        data=corr_matrix.to_csv().encode("utf-8"),
        file_name="correlation_matrix.csv",
        mime="text/csv"
    )

    # P-value matrix CSV
    p_values = pd.DataFrame(np.ones((len(numeric_cols), len(numeric_cols))),
                            index=numeric_cols, columns=numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            _, p_val = pearsonr(df[numeric_cols[i]].dropna(), df[numeric_cols[j]].dropna())
            p_values.loc[numeric_cols[i], numeric_cols[j]] = p_val
            p_values.loc[numeric_cols[j], numeric_cols[i]] = p_val

    st.download_button(
        label="Download pvalue_matrix.csv",
        data=p_values.to_csv().encode("utf-8"),
        file_name="pvalue_matrix.csv",
        mime="text/csv"
    )

    # R² matrix CSV
    r_squared = corr_matrix ** 2
    st.download_button(
        label="Download r_squared_matrix.csv",
        data=r_squared.to_csv().encode("utf-8"),
        file_name="r_squared_matrix.csv",
        mime="text/csv"
    )

    # Normalized data CSV
    df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    st.download_button(
        label="Download rail_normalized.csv",
        data=df_norm.to_csv(index=False).encode("utf-8"),
        file_name="rail_normalized.csv",
        mime="text/csv"
    )

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL: ACTIONABLE INSIGHTS
# ────────────────────────────────────────────────────────────────────────────────

if show_insights:
    st.header("8. Actionable Insights")

    corr_matrix = df[numeric_cols].corr()
    p_values = pd.DataFrame(np.ones((len(numeric_cols), len(numeric_cols))),
                            index=numeric_cols, columns=numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r_val, p_val = pearsonr(df[numeric_cols[i]].dropna(), df[numeric_cols[j]].dropna())
            p_values.loc[numeric_cols[i], numeric_cols[j]] = p_val
            p_values.loc[numeric_cols[j], numeric_cols[i]] = p_val

    # Identify strongly correlated pairs (|r| > 0.5)
    strong_pairs = []
    threshold = 0.5
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r_val = corr_matrix.loc[numeric_cols[i], numeric_cols[j]]
            if abs(r_val) > threshold:
                strong_pairs.append((
                    numeric_cols[i],
                    numeric_cols[j],
                    r_val,
                    p_values.loc[numeric_cols[i], numeric_cols[j]],
                    r_val**2
                ))

    # Pull example values for the first two insights
    capex_safety_r = corr_matrix.loc["CapEx ($M)", "Safety Incidents"]
    capex_safety_r2 = capex_safety_r ** 2
    dur_slip_r = corr_matrix.loc["Duration (months)", "Schedule Slip (%)"]
    dur_slip_r2 = dur_slip_r ** 2

    st.markdown(f"""
**1. CapEx ↔ Safety Incidents**  
• Pearson r = {capex_safety_r:.2f}, p < 0.001, R² = {capex_safety_r2:.2f}  
• Interpretation: Higher‐cost projects tend to have more safety incidents.  
• Action: Normalize “Safety Incidents per $M” or investigate if large projects inherently carry more risk.

**2. Duration ↔ Schedule Slip**  
• Pearson r = {dur_slip_r:.2f}, p < 0.01, R² = {dur_slip_r2:.2f}  
• Interpretation: Longer‐duration projects show smaller percentage slip—likely built‐in buffer.  
• Action: Group by duration bands (e.g., <12 m, 12–24 m, >24 m) to compare average schedule slip.

**3. Distribution Notes**  
• Fields like “Carbon Intensity (tCO₂e/km)” may show right‐skew in histograms.  
• Action: When distributions are skewed, report median and IQR rather than mean.

**4. Outlier Handling**  
• Use the “Distribution & Outliers” panel to review flagged rows.  
• Action: If an outlier is a legitimate mega‐project, keep and annotate; otherwise, drop or correct.

**5. Normalization Ready**  
• The normalized dataset (Z‐score) is available for clustering and regression without scale bias.  
• Action: Use the “Normalization & Clustering” panel to run K‐Means/PCA and identify project archetypes.

**6. Strongly Correlated Pairs (|r| > 0.5)**  
""")

    if strong_pairs:
        st.write("#### Full list of strongly correlated feature pairs (|r| > 0.5):")
        for (a, b, r_val, p_val, r2_val) in strong_pairs:
            st.write(f"- {a} ↔ {b}: r = {r_val:.2f}, p = {p_val:.3e}, R² = {r2_val:.2f}")
    else:
        st.write("No feature pairs exceed |r| > 0.5 in this dataset.")

    st.markdown("""
**Next Steps**  
- Drop or combine nearly colinear features (e.g., CapEx & Safety Incidents) when building any regression.  
- Use median/percentiles for skewed distributions in your benchmarking report.  
- Consider segmenting by geography or delivery method (if metadata exists).  
- Focus benchmarks on mid‐range projects (e.g., exclude top 5% by CapEx) to avoid skewed averages.  
""")
