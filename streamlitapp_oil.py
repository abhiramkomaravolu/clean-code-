import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix

# Optional: clustering/PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ────────────────────────────────────────────────────────────────────────────────
# 1) LOAD & CLEAN DATA
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_and_clean():
    df = pd.read_csv("synthetic_oil_projects.csv")

    # 1.1) Parse “Scope (km or m3)” into numeric value + unit
    def parse_scope(val):
        if pd.isna(val):
            return (np.nan, None)
        parts = val.split()
        if len(parts) == 2:
            num_str, unit = parts
            try:
                num = float(num_str.replace(",", ""))
            except:
                num = np.nan
            return (num, unit)
        else:
            return (np.nan, None)

    scope_parsed = df["Scope (km or m3)"].apply(parse_scope)
    df["Scope_Value"] = scope_parsed.apply(lambda x: x[0])
    df["Scope_Unit"] = scope_parsed.apply(lambda x: x[1])

    # 1.2) Convert “Delay %” and “Rework Cost %” into numeric floats
    for col in ["Delay %", "Rework Cost %"]:
        if col in df.columns:
            df[col + "_Num"] = (
                df[col]
                .astype(str)
                .str.rstrip("%")
                .replace("", np.nan)
                .astype(float)
            )

    return df

df = load_and_clean()

# 1.3) Define numeric and categorical columns
numeric_cols = []
for c in ["CapEx ($M)", "Duration (months)", "Safety Score", "Scope_Value", "Delay %_Num", "Rework Cost %_Num"]:
    if c in df.columns:
        numeric_cols.append(c)

cat_cols = []
for c in ["Project ID", "Location", "Type", "Scope_Unit"]:
    if c in df.columns:
        cat_cols.append(c)

# ────────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Oil Projects Benchmarking")
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
st.sidebar.write("Numeric fields detected:")
for col in numeric_cols:
    st.sidebar.write(f"• {col}")
if cat_cols:
    st.sidebar.write("Categorical fields detected:")
    for col in cat_cols:
        st.sidebar.write(f"• {col}")

# ────────────────────────────────────────────────────────────────────────────────
# 3) PANEL: DASHBOARD OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_dashboard:
    st.header("0. Dashboard Overview")
    st.markdown("High-level KPIs and quick visual summaries for the oil projects dataset.")

    total_projects = df.shape[0]
    avg_capex      = df["CapEx ($M)"].mean()          if "CapEx ($M)" in df.columns else np.nan
    med_duration   = df["Duration (months)"].median() if "Duration (months)" in df.columns else np.nan
    avg_safety     = df["Safety Score"].mean()        if "Safety Score" in df.columns else np.nan
    avg_delay      = df["Delay %_Num"].mean()         if "Delay %_Num" in df.columns else np.nan
    avg_rework     = df["Rework Cost %_Num"].mean()   if "Rework Cost %_Num" in df.columns else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Projects", f"{int(total_projects)}")
    c2.metric("Avg CapEx ($M)", f"{avg_capex:,.1f}")
    c3.metric("Median Duration (m)", f"{med_duration:.1f}")
    c4.metric("Avg Safety Score", f"{avg_safety:.1f}")
    c5.metric("Avg Delay (%)", f"{avg_delay:.1f}" if not np.isnan(avg_delay) else "N/A")
    c6.metric("Avg Rework Cost (%)", f"{avg_rework:.1f}" if not np.isnan(avg_rework) else "N/A")

    st.markdown("---")

    # 3.1) Projects by Type (bar)
    if "Type" in df.columns:
        st.subheader("Projects by Type")
        type_counts = df["Type"].value_counts()
        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.bar(type_counts.index, type_counts.values, color="teal")
        ax_t.set_xlabel("Type")
        ax_t.set_ylabel("Number of Projects")
        ax_t.set_title("Projects per Type")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_t)

    # 3.2) Avg CapEx by Type (bar)
    if "Type" in df.columns and "CapEx ($M)" in df.columns:
        st.subheader("Avg CapEx by Type")
        capex_by_type = df.groupby("Type")["CapEx ($M)"].mean().sort_values(ascending=False)
        fig_ct, ax_ct = plt.subplots(figsize=(6, 3))
        ax_ct.bar(capex_by_type.index, capex_by_type.values, color="orange")
        ax_ct.set_xlabel("Type")
        ax_ct.set_ylabel("Avg CapEx ($M)")
        ax_ct.set_title("Avg CapEx per Type")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_ct)

    st.markdown("---")

    # 3.3) Scope Unit Distribution (pie)
    if "Scope_Unit" in df.columns:
        st.subheader("Scope Unit Distribution")
        unit_counts = df["Scope_Unit"].value_counts()
        fig_u, ax_u = plt.subplots(figsize=(5, 3))
        ax_u.pie(unit_counts.values, labels=unit_counts.index, autopct="%1.1f%%", startangle=140)
        ax_u.set_title("Scope Unit Breakdown")
        st.pyplot(fig_u)

    st.markdown("---")

    # 3.4) Top 5 Projects by CapEx
    if "CapEx ($M)" in df.columns:
        st.subheader("Top 5 Projects by CapEx ($M)")
        top5_capex = df.sort_values("CapEx ($M)", ascending=False).head(5)
        st.table(top5_capex[["Project ID", "CapEx ($M)", "Type", "Location"]].reset_index(drop=True))

    # 3.5) Correlation Snapshot (mini‐heatmap)
    if len(numeric_cols) >= 2:
        st.markdown("---")
        st.subheader("Correlation Snapshot")
        corr_sm = df[numeric_cols].corr()
        fig_cs, ax_cs = plt.subplots(figsize=(4, 4))
        im = ax_cs.imshow(corr_sm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_cs.set_xticks(range(len(numeric_cols)))
        ax_cs.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
        ax_cs.set_yticks(range(len(numeric_cols)))
        ax_cs.set_yticklabels(numeric_cols, fontsize=8)
        fig_cs.colorbar(im, ax=ax_cs, fraction=0.046, pad=0.04)
        st.pyplot(fig_cs)

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_overview:
    st.header("1. Data Overview")
    st.write("First five rows of the dataset:")
    st.dataframe(df.head())

    st.write("### Descriptive Statistics (Numeric Columns)")
    st.dataframe(df[numeric_cols].describe())

    if cat_cols:
        st.write("### Value Counts (Categorical Columns)")
        for c in cat_cols:
            st.write(f"**{c}**")
            vc = df[c].value_counts().reset_index().rename(columns={"index": c, c: "Count"})
            st.dataframe(vc)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL: DISTRIBUTION & OUTLIER ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────

if show_distributions:
    st.header("2. Distribution & Outlier Analysis")

    # 5.1) Numeric distribution & outliers
    col = st.selectbox("Select a numeric column:", numeric_cols)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax.set_title(f"Histogram of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()

    st.write(f"**Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}**")
    st.write(f"Outlier bounds: [{lower:.2f}, {upper:.2f}]")
    st.write(f"Number of outliers in '{col}': {outlier_count}")

    if st.checkbox("Show sample outlier rows for this column"):
        st.dataframe(df.loc[outlier_mask].head(10))

    if st.checkbox("Show boxplots for all numeric columns"):
        fig_box, ax_box = plt.subplots(figsize=(10, 4))
        ax_box.boxplot([df[c].dropna() for c in numeric_cols], labels=numeric_cols, vert=False)
        ax_box.set_title("Boxplot of All Numeric Columns")
        st.pyplot(fig_box)

    st.markdown("---")

    # 5.2) Pie chart for categorical distribution
    if cat_cols:
        st.subheader("Categorical Distribution (Pie Chart)")
        cat = st.selectbox("Select a categorical column:", cat_cols)
        counts = df[cat].value_counts()
        fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
        ax_pie.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=140)
        ax_pie.set_title(f"{cat} Distribution")
        st.pyplot(fig_pie)

# ────────────────────────────────────────────────────────────────────────────────
# 6) PANEL: CORRELATION & SIGNIFICANCE
# ────────────────────────────────────────────────────────────────────────────────

if show_corr:
    st.header("3. Correlation & Significance")

    corr_matrix = df[numeric_cols].corr()
    st.write("### Pearson Correlation Matrix")
    st.dataframe(corr_matrix)

    if st.checkbox("Show correlation heatmap"):
        fig_ch, ax_ch = plt.subplots(figsize=(6, 6))
        im = ax_ch.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_ch.set_xticks(range(len(numeric_cols)))
        ax_ch.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax_ch.set_yticks(range(len(numeric_cols)))
        ax_ch.set_yticklabels(numeric_cols)
        fig_ch.colorbar(im, ax=ax_ch, fraction=0.046, pad=0.04)
        st.pyplot(fig_ch)

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

    fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
    ax_sc.scatter(df[x_col], df[y_col], alpha=0.6)
    ax_sc.set_xlabel(x_col)
    ax_sc.set_ylabel(y_col)
    ax_sc.set_title(f"{x_col} vs. {y_col}")
    st.pyplot(fig_sc)

    r_val, p_val = pearsonr(df[x_col].dropna(), df[y_col].dropna())
    st.write(f"Pearson r = {r_val:.2f}, p-value = {p_val:.3e}, R² = {r_val**2:.2f}")

    if st.checkbox("Show full scatter matrix for selected columns"):
        cols_sel = st.multiselect("Pick ≥2 columns:", numeric_cols, default=numeric_cols[:3])
        if len(cols_sel) >= 2:
            fig_sm2 = scatter_matrix(df[cols_sel], alpha=0.6, diagonal="hist", figsize=(8, 8))
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
            file_name="oil_data_normalized.csv",
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
        fig_pca, ax_pca = plt.subplots(figsize=(5, 4))
        ax_pca.scatter(pcs[:, 0], pcs[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.7)
        ax_pca.set_xlabel("PC 1")
        ax_pca.set_ylabel("PC 2")
        ax_pca.set_title("PCA Projection by K-Means Cluster")
        st.pyplot(fig_pca)

# ────────────────────────────────────────────────────────────────────────────────
# 9) PANEL: TOP 10 PROJECTS
# ────────────────────────────────────────────────────────────────────────────────

if show_top10:
    st.header("6. Top 10 Projects by Chosen Metric")

    metric = st.selectbox(
        "Select metric for ranking:", numeric_cols,
        index=numeric_cols.index("CapEx ($M)") if "CapEx ($M)" in numeric_cols else 0
    )
    order = st.radio("Sort order:", ["Descending", "Ascending"])
    ascending = (order == "Ascending")

    df_sorted = df.sort_values(by=metric, ascending=ascending).head(10)
    st.write(f"Top 10 projects by **{metric}** ({order}):")
    st.dataframe(df_sorted.reset_index(drop=True))

    st.subheader(f"Top 10 Project IDs vs. {metric}")
    fig_t10, ax_t10 = plt.subplots(figsize=(6, 4))
    ax_t10.barh(df_sorted["Project ID"].astype(str), df_sorted[metric], color="skyblue")
    ax_t10.set_xlabel(metric)
    ax_t10.set_ylabel("Project ID")
    ax_t10.invert_yaxis()
    ax_t10.set_title(f"Top 10 by {metric}")
    st.pyplot(fig_t10)

    if st.checkbox("Rank by composite score (CapEx × Safety Score)"):
        df_comp = df.copy()
        if "CapEx ($M)" in df_comp.columns and "Safety Score" in df_comp.columns:
            df_comp["Composite Score"] = df_comp["CapEx ($M)"] * df_comp["Safety Score"]
            df_cs = df_comp.sort_values("Composite Score", ascending=False).head(10)
            st.write("### Top 10 by Composite Score")
            st.dataframe(df_cs[["Project ID", "CapEx ($M)", "Safety Score", "Composite Score"]].reset_index(drop=True))

            fig_cs2, ax_cs2 = plt.subplots(figsize=(6, 4))
            ax_cs2.barh(df_cs["Project ID"].astype(str), df_cs["Composite Score"], color="salmon")
            ax_cs2.set_xlabel("Composite Score")
            ax_cs2.set_ylabel("Project ID")
            ax_cs2.invert_yaxis()
            ax_cs2.set_title("Top 10 by Composite Score")
            st.pyplot(fig_cs2)
        else:
            st.write("Composite Score requires both `CapEx ($M)` and `Safety Score`.")

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

if show_downloads:
    st.header("7. Download Reports")

    desc = df[numeric_cols].describe()
    st.download_button("Download descriptive_statistics.csv", desc.to_csv().encode("utf-8"), "descriptive_statistics.csv", "text/csv")

    corr_matrix = df[numeric_cols].corr()
    st.download_button("Download correlation_matrix.csv", corr_matrix.to_csv().encode("utf-8"), "correlation_matrix.csv", "text/csv")

    # p-value matrix
    def compute_pvalues(df_num):
        cols = df_num.columns
        pmat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r_val, p_val = pearsonr(df_num[cols[i]].dropna(), df_num[cols[j]].dropna())
                pmat.loc[cols[i], cols[j]] = p_val
                pmat.loc[cols[j], cols[i]] = p_val
        return pmat

    p_values = compute_pvalues(df[numeric_cols])
    st.download_button("Download pvalue_matrix.csv", p_values.to_csv().encode("utf-8"), "pvalue_matrix.csv", "text/csv")

    r_squared = corr_matrix ** 2
    st.download_button("Download r_squared_matrix.csv", r_squared.to_csv().encode("utf-8"), "r_squared_matrix.csv", "text/csv")

    df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    st.download_button("Download oil_data_normalized.csv", df_norm.to_csv(index=False).encode("utf-8"), "oil_data_normalized.csv", "text/csv")

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
            p_val = p_values.loc[numeric_cols[i], numeric_cols[j]]
            if abs(r_val) > threshold and p_val < 0.05:
                strong_pairs.append((numeric_cols[i], numeric_cols[j], r_val, p_val, r_val**2))

    insight_md = ""
    if strong_pairs:
        insight_md += "**Strong correlations (|r| > 0.5 & p < 0.05)**  \n"
        for a, b, r_val, p_val, r2 in strong_pairs:
            insight_md += f"- {a} ↔ {b}: r = {r_val:.2f}, p = {p_val:.3e}, R² = {r2:.2f}  \n"
    else:
        insight_md += "No numeric feature pairs exceed |r| > 0.5 with p < 0.05.  \n\n"

    insight_md += "**Outlier Summary**  \n"
    # Compute outlier counts for each numeric column
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = (df[numeric_cols] < lower) | (df[numeric_cols] > upper)
    outlier_counts = outlier_mask.sum()
    for col in numeric_cols:
        insight_md += f"- {col}: {outlier_counts[col]} outliers flagged  \n"

    insight_md += "\n**Next Steps**  \n"
    insight_md += "- Use histograms/boxplots to decide if medians or transformations are needed.  \n"
    insight_md += "- Review high‐impact features (e.g. CapEx ↔ Safety Score) before modeling.  \n"
    insight_md += "- Consider segmenting by “Type” or “Location” for more granular benchmarks.  \n"
    insight_md += "- Use normalized data (`oil_data_normalized.csv`) for clustering/regression.  \n"

    st.markdown(insight_md)
