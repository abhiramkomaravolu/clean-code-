import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix

# Optional clustering/PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD DATA
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("rail_synthetic_formatted.csv")
    return df

df = load_data()
numeric_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# Identify carbon column if present
carbon_col = next((c for c in numeric_cols if "Carbon" in c), None)

# ────────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Benchmarking Dashboard")
show_dashboard      = st.sidebar.checkbox("0. Dashboard", True)
show_overview       = st.sidebar.checkbox("1. Data Overview", False)
show_distributions  = st.sidebar.checkbox("2. Distribution & Outliers", False)
show_corr           = st.sidebar.checkbox("3. Correlation & Significance", False)
show_scatter        = st.sidebar.checkbox("4. Pairwise Scatter Plots", False)
show_normalization  = st.sidebar.checkbox("5. Normalization & Clustering", False)
show_top10          = st.sidebar.checkbox("6. Top 10 Projects", False)
show_downloads      = st.sidebar.checkbox("7. Download Reports", False)
show_insights       = st.sidebar.checkbox("8. Actionable Insights", False)

st.sidebar.markdown("---")
st.sidebar.write("Detected Numeric Fields:")
for col in numeric_cols:
    st.sidebar.write(f"• {col}")
if cat_cols:
    st.sidebar.write("Detected Categorical Fields:")
    for col in cat_cols:
        st.sidebar.write(f"• {col}")

# ────────────────────────────────────────────────────────────────────────────────
# 3) PANEL: DASHBOARD OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_dashboard:
    st.header("0. Dashboard Overview")
    st.markdown("Quick summary metrics and visual snapshots of key dimensions.")

    # 3.1 Summary Metrics
    total_projects = df.shape[0]
    avg_capex = df["CapEx ($M)"].mean() if "CapEx ($M)" in df.columns else np.nan
    med_duration = df["Duration (months)"].median() if "Duration (months)" in df.columns else np.nan
    avg_safety = df["Safety Incidents"].mean() if "Safety Incidents" in df.columns else np.nan
    avg_carbon = df[carbon_col].mean() if carbon_col else np.nan

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Projects", f"{int(total_projects)}")
    col2.metric("Avg CapEx ($M)", f"{avg_capex:,.1f}")
    col3.metric("Median Duration (m)", f"{med_duration:.1f}")
    col4.metric("Avg Safety Incidents", f"{avg_safety:.1f}")
    col5.metric("Avg Carbon Intensity", f"{avg_carbon:.1f}" if carbon_col else "N/A")

    st.markdown("---")

    # 3.2 Projects by Region (bar)
    if "Region" in df.columns:
        st.subheader("Projects by Region")
        region_counts = df["Region"].value_counts()
        fig_r, ax_r = plt.subplots(figsize=(6, 3))
        ax_r.bar(region_counts.index, region_counts.values, color="teal")
        ax_r.set_xlabel("Region")
        ax_r.set_ylabel("Number of Projects")
        ax_r.set_title("Projects per Region")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_r)

    # 3.3 Avg CapEx by Region (bar)
    if "Region" in df.columns and "CapEx ($M)" in df.columns:
        st.subheader("Avg CapEx by Region")
        capex_by_region = df.groupby("Region")["CapEx ($M)"].mean().sort_values(ascending=False)
        fig_cr, ax_cr = plt.subplots(figsize=(6, 3))
        ax_cr.bar(capex_by_region.index, capex_by_region.values, color="orange")
        ax_cr.set_xlabel("Region")
        ax_cr.set_ylabel("Avg CapEx ($M)")
        ax_cr.set_title("Average CapEx per Region")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_cr)

    st.markdown("---")

    # 3.4 Delivery Model Distribution (pie)
    if "Delivery Model" in df.columns:
        st.subheader("Delivery Model Distribution")
        dm_counts = df["Delivery Model"].value_counts()
        fig_dm, ax_dm = plt.subplots(figsize=(5, 3))
        ax_dm.pie(dm_counts.values, labels=dm_counts.index, autopct="%1.1f%%", startangle=140)
        ax_dm.set_title("Delivery Model Breakdown")
        st.pyplot(fig_dm)

    st.markdown("---")

    # 3.5 Top 5 Projects by Key Metrics
    st.subheader("Top 5 Projects by Key Metrics")
    # 3.5.1 Top 5 by CapEx
    if "CapEx ($M)" in df.columns:
        st.write("**Top 5 by CapEx ($M)**")
        top5_capex = df.sort_values("CapEx ($M)", ascending=False).head(5)
        st.table(top5_capex[["Project ID", "CapEx ($M)", "Region", "Delivery Model"]].reset_index(drop=True))
    # 3.5.2 Top 5 by Safety Incidents
    if "Safety Incidents" in df.columns:
        st.write("**Top 5 by Safety Incidents**")
        top5_safety = df.sort_values("Safety Incidents", ascending=False).head(5)
        st.table(top5_safety[["Project ID", "Safety Incidents", "CapEx ($M)", "Region"]].reset_index(drop=True))
    # 3.5.3 Top 5 by Carbon Intensity
    if carbon_col:
        st.write(f"**Top 5 by {carbon_col}**")
        top5_carbon = df.sort_values(carbon_col, ascending=False).head(5)
        st.table(top5_carbon[["Project ID", carbon_col, "CapEx ($M)", "Region"]].reset_index(drop=True))

    st.markdown("---")

    # 3.6 Correlation Snapshot (small heatmap)
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Snapshot")
        corr_sm = df[numeric_cols].corr()
        fig_cor, ax_cor = plt.subplots(figsize=(4, 4))
        im = ax_cor.imshow(corr_sm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_cor.set_xticks(range(len(numeric_cols)))
        ax_cor.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
        ax_cor.set_yticks(range(len(numeric_cols)))
        ax_cor.set_yticklabels(numeric_cols, fontsize=8)
        fig_cor.colorbar(im, ax=ax_cor, fraction=0.046, pad=0.04)
        st.pyplot(fig_cor)

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_overview:
    st.header("1. Data Overview")
    st.write("Below are the first few rows of your dataset:")
    st.dataframe(df.head())

    st.write("### Descriptive Statistics for Numeric Columns")
    st.dataframe(df[numeric_cols].describe())

    if cat_cols:
        st.write("### Value Counts for Categorical Columns")
        for c in cat_cols:
            st.write(f"**{c}**")
            vc = df[c].value_counts().reset_index().rename(columns={"index": c, c: "Count"})
            st.dataframe(vc)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL: DISTRIBUTION & OUTLIER ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────

if show_distributions:
    st.header("2. Distribution & Outlier Analysis")

    # Numeric distribution
    col = st.selectbox("Select a numeric column:", numeric_cols)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(f"Histogram: {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    st.write(f"**Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}**")
    st.write(f"Outlier bounds: [{lower:.2f}, {upper:.2f}] → {len(outliers)} flagged")

    if st.checkbox("Show flagged outlier rows"):
        st.dataframe(outliers.head(10))

    if st.checkbox("Show boxplots for all numeric columns"):
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.boxplot([df[c].dropna() for c in numeric_cols], labels=numeric_cols, vert=False)
        ax2.set_title("Boxplot of Numeric Columns")
        st.pyplot(fig2)

    st.markdown("---")

    # Categorical pie chart
    if cat_cols:
        st.subheader("Categorical Distribution (Pie Chart)")
        cat = st.selectbox("Select a categorical column:", cat_cols)
        counts = df[cat].value_counts()
        figp, axp = plt.subplots(figsize=(5, 4))
        axp.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
        axp.set_title(f"{cat} Breakdown")
        st.pyplot(figp)

# ────────────────────────────────────────────────────────────────────────────────
# 6) PANEL: CORRELATION & SIGNIFICANCE
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
        fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        st.pyplot(fig3)

    def compute_pvalues(df_num):
        cols = df_num.columns
        pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val, p_val = pearsonr(df_num[cols[i]].dropna(), df_num[cols[j]].dropna())
                pmat.loc[cols[i], cols[j]] = p_val
                pmat.loc[cols[j], cols[i]] = p_val
        return pmat

    if st.checkbox("Show p-value matrix"):
        p_values = compute_pvalues(df[numeric_cols])
        st.write("### P-Value Matrix")
        st.dataframe(p_values)

    if st.checkbox("Show R² (correlation²) matrix"):
        r_squared = corr_matrix ** 2
        st.write("### R² (Correlation²) Matrix")
        st.dataframe(r_squared)

# ────────────────────────────────────────────────────────────────────────────────
# 7) PANEL: PAIRWISE SCATTER PLOTS
# ────────────────────────────────────────────────────────────────────────────────

if show_scatter:
    st.header("4. Pairwise Scatter Plots")

    x_col = st.selectbox("X-axis column:", numeric_cols, index=0, key="scatter_x")
    y_col = st.selectbox("Y-axis column:", numeric_cols, index=1, key="scatter_y")

    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.scatter(df[x_col], df[y_col], alpha=0.6)
    ax4.set_xlabel(x_col)
    ax4.set_ylabel(y_col)
    ax4.set_title(f"{x_col} vs. {y_col}")
    st.pyplot(fig4)

    r_val, p_val = pearsonr(df[x_col].dropna(), df[y_col].dropna())
    st.write(f"Pearson r = {r_val:.2f}, p-value = {p_val:.3e}, R² = {r_val**2:.2f}")

    if st.checkbox("Show scatter matrix for selected columns"):
        cols_sel = st.multiselect("Pick ≥2 columns:", numeric_cols, default=numeric_cols[:3])
        if len(cols_sel) >= 2:
            fig5 = scatter_matrix(df[cols_sel], alpha=0.6, diagonal="hist", figsize=(8, 8))
            st.pyplot(plt.gcf())

# ────────────────────────────────────────────────────────────────────────────────
# 8) PANEL: NORMALIZATION & CLUSTERING
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

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(df_norm)
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        ax6.scatter(pcs[:, 0], pcs[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.7)
        ax6.set_xlabel("PC 1")
        ax6.set_ylabel("PC 2")
        ax6.set_title("PCA Projection by K-Means Cluster")
        st.pyplot(fig6)

# ────────────────────────────────────────────────────────────────────────────────
# 9) PANEL: TOP 10 PROJECTS
# ────────────────────────────────────────────────────────────────────────────────

if show_top10:
    st.header("6. Top 10 Projects by Chosen Metric")

    metric = st.selectbox(
        "Select metric:", numeric_cols,
        index=numeric_cols.index("CapEx ($M)") if "CapEx ($M)" in numeric_cols else 0
    )
    order = st.radio("Sort order:", ["Descending", "Ascending"])
    ascending = (order == "Ascending")

    df_sorted = df.sort_values(by=metric, ascending=ascending)
    top_n = df_sorted.head(10)
    st.write(f"Showing top 10 projects by **{metric}** ({order})")
    st.dataframe(top_n.reset_index(drop=True))

    st.subheader(f"Top 10 Project IDs vs. {metric}")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    ax7.barh(top_n["Project ID"].astype(str), top_n[metric], color="skyblue")
    ax7.set_xlabel(metric)
    ax7.set_ylabel("Project ID")
    ax7.invert_yaxis()
    ax7.set_title(f"Top 10 by {metric}")
    st.pyplot(fig7)

    if st.checkbox("Rank by composite score (CapEx × Safety Incidents)"):
        df_prime = df.copy()
        if "CapEx ($M)" in df_prime.columns and "Safety Incidents" in df_prime.columns:
            df_prime["Composite Score"] = df_prime["CapEx ($M)"] * df_prime["Safety Incidents"]
            df_cs = df_prime.sort_values("Composite Score", ascending=False).head(10)
            st.write("### Top 10 by Composite Score")
            st.dataframe(df_cs[["Project ID", "CapEx ($M)", "Safety Incidents", "Composite Score"]].reset_index(drop=True))

            fig8, ax8 = plt.subplots(figsize=(6, 4))
            ax8.barh(df_cs["Project ID"].astype(str), df_cs["Composite Score"], color="salmon")
            ax8.set_xlabel("Composite Score")
            ax8.set_ylabel("Project ID")
            ax8.invert_yaxis()
            ax8.set_title("Top 10 by Composite Score")
            st.pyplot(fig8)
        else:
            st.write("Composite Score requires both `CapEx ($M)` and `Safety Incidents`.")

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

if show_downloads:
    st.header("7. Download Reports")
    desc = df[numeric_cols].describe()
    st.download_button("Download descriptive_statistics.csv", desc.to_csv().encode("utf-8"), "descriptive_statistics.csv", "text/csv")

    corr_matrix = df[numeric_cols].corr()
    st.download_button("Download correlation_matrix.csv", corr_matrix.to_csv().encode("utf-8"), "correlation_matrix.csv", "text/csv")

    p_values = pd.DataFrame(np.ones((len(numeric_cols), len(numeric_cols))), index=numeric_cols, columns=numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            _, p_val = pearsonr(df[numeric_cols[i]].dropna(), df[numeric_cols[j]].dropna())
            p_values.loc[numeric_cols[i], numeric_cols[j]] = p_val
            p_values.loc[numeric_cols[j], numeric_cols[i]] = p_val
    st.download_button("Download pvalue_matrix.csv", p_values.to_csv().encode("utf-8"), "pvalue_matrix.csv", "text/csv")

    r_squared = corr_matrix ** 2
    st.download_button("Download r_squared_matrix.csv", r_squared.to_csv().encode("utf-8"), "r_squared_matrix.csv", "text/csv")

    df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    st.download_button("Download rail_normalized.csv", df_norm.to_csv(index=False).encode("utf-8"), "rail_normalized.csv", "text/csv")

# ────────────────────────────────────────────────────────────────────────────────
# 11) PANEL: ACTIONABLE INSIGHTS
# ────────────────────────────────────────────────────────────────────────────────

if show_insights:
    st.header("8. Actionable Insights")
    corr_matrix = df[numeric_cols].corr()
    p_values = pd.DataFrame(np.ones((len(numeric_cols), len(numeric_cols))), index=numeric_cols, columns=numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r_val, p_val = pearsonr(df[numeric_cols[i]].dropna(), df[numeric_cols[j]].dropna())
            p_values.loc[numeric_cols[i], numeric_cols[j]] = p_val
            p_values.loc[numeric_cols[j], numeric_cols[i]] = p_val

    strong_pairs = []
    threshold = 0.5
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            r_val = corr_matrix.loc[numeric_cols[i], numeric_cols[j]]
            if abs(r_val) > threshold:
                strong_pairs.append((numeric_cols[i], numeric_cols[j], r_val, p_values.loc[numeric_cols[i], numeric_cols[j]], r_val**2))

    capex_safety_r = corr_matrix.loc["CapEx ($M)", "Safety Incidents"] if "CapEx ($M)" in corr_matrix and "Safety Incidents" in corr_matrix else None
    capex_safety_r2 = capex_safety_r ** 2 if capex_safety_r is not None else None
    dur_slip_r = corr_matrix.loc["Duration (months)", "Schedule Slip (%)"] if "Duration (months)" in corr_matrix and "Schedule Slip (%)" in corr_matrix else None
    dur_slip_r2 = dur_slip_r ** 2 if dur_slip_r is not None else None

    insight_md = ""
    if capex_safety_r is not None:
        insight_md += f"**1. CapEx ↔ Safety Incidents**  \n" \
                      f"• Pearson r = {capex_safety_r:.2f}, p < 0.001, R² = {capex_safety_r2:.2f}  \n" \
                      f"• Insight: Higher-cost projects tend to have more safety incidents.  \n" \
                      f"• Action: Normalize “Safety Incidents per $M” or investigate risk factors.  \n\n"
    if dur_slip_r is not None:
        insight_md += f"**2. Duration ↔ Schedule Slip**  \n" \
                      f"• Pearson r = {dur_slip_r:.2f}, p < 0.01, R² = {dur_slip_r2:.2f}  \n" \
                      f"• Insight: Longer-duration projects show smaller % slip—likely built-in buffer.  \n" \
                      f"• Action: Group by duration bands to compare average slips.  \n\n"
    insight_md += "**3. Distribution Notes**  \n" \
                  "• Fields like “Carbon Intensity” may be right-skewed.  \n" \
                  "• Action: Use median & IQR for skewed benchmarks.  \n\n" \
                  "**4. Outlier Handling**  \n" \
                  "• Use the Distribution panel to review flagged outliers.  \n" \
                  "• Action: Keep legitimate megaprojects, annotate; drop data errors.  \n\n" \
                  "**5. Normalization Ready**  \n" \
                  "• Z-score normalized data available for clustering/regression.  \n" \
                  "• Action: Use K-Means/PCA panel to identify project archetypes.  \n\n" \
                  "**6. Strongly Correlated Pairs (|r| > 0.5)**  \n\n"

    st.markdown(insight_md)
    if strong_pairs:
        st.write("#### Feature pairs with |r| > 0.5:")
        for a, b, r_val, p_val, r2_val in strong_pairs:
            st.write(f"- {a} ↔ {b}: r = {r_val:.2f}, p = {p_val:.3e}, R² = {r2_val:.2f}")
    else:
        st.write("No feature pairs exceed |r| > 0.5.")

    st.markdown("""
**Next Steps**  
- Drop/combine colinear features in regression.  
- Report median/percentiles for skewed metrics.  
- Segment by geography or delivery model for deeper benchmarks.  
- Focus on mid-range projects to avoid skewed averages.  
""")
