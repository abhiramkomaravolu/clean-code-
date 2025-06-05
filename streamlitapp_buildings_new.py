import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

@st.cache_data
def load_data():
    return pd.read_excel("buildings_synthetic_output.xlsx")

df = load_data()

# 1) Define numeric and categorical columns
numeric_cols = ["GFA (m²)", "Height", "CapEx ($M)"]
cat_cols_all = ["Project ID", "Type"]

# For pie‐charts in Distribution panel, exclude Project ID
pie_cols = [c for c in cat_cols_all if c != "Project ID"]  # → ["Type"]

# ────────────────────────────────────────────────────────────────────────────────
# 2) SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────────────────────────

st.sidebar.title("Buildings Benchmarking")
show_dash     = st.sidebar.checkbox("0. Dashboard", True)
show_overview = st.sidebar.checkbox("1. Data Overview", False)
show_dist     = st.sidebar.checkbox("2. Distribution & Outliers", False)
show_corr     = st.sidebar.checkbox("3. Correlation & Significance", False)
show_scatter  = st.sidebar.checkbox("4. Pairwise Scatter Plots", False)
show_norm     = st.sidebar.checkbox("5. Normalization & Clustering", False)
show_top10    = st.sidebar.checkbox("6. Top 10", False)
show_dl       = st.sidebar.checkbox("7. Download Reports", False)
show_insights = st.sidebar.checkbox("8. Actionable Insights", False)

st.sidebar.markdown("---")
st.sidebar.write("Detected Numeric Fields:")
for col in numeric_cols:
    st.sidebar.write(f"• {col}")
st.sidebar.write("Detected Categorical Fields:")
for col in cat_cols_all:
    st.sidebar.write(f"• {col}")

# ────────────────────────────────────────────────────────────────────────────────
# 3) PANEL: DASHBOARD OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_dash:
    st.header("0. Dashboard Overview")
    total_projects = df.shape[0]
    avg_gfa        = df["GFA (m²)"].mean()
    med_height     = df["Height"].median()
    avg_capex      = df["CapEx ($M)"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Projects", f"{total_projects}")
    c2.metric("Avg GFA (m²)", f"{avg_gfa:,.0f}")
    c3.metric("Avg CapEx ($M)", f"{avg_capex:,.1f}")

    st.markdown("---")
    if "Type" in df.columns:
        st.subheader("Projects by Type")
        type_counts = df["Type"].value_counts()
        fig_t, ax_t = plt.subplots(figsize=(6, 3))
        ax_t.bar(type_counts.index, type_counts.values, color="teal")
        ax_t.set_xlabel("Type")
        ax_t.set_ylabel("Count")
        ax_t.set_title("Projects per Type")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_t)

# ────────────────────────────────────────────────────────────────────────────────
# 4) PANEL: DATA OVERVIEW
# ────────────────────────────────────────────────────────────────────────────────

if show_overview:
    st.header("1. Data Overview")
    st.dataframe(df.head())

    st.write("### Descriptive Statistics")
    st.dataframe(df[numeric_cols].describe())

    if cat_cols_all:
        st.write("### Categorical Value Counts")
        for c in cat_cols_all:
            st.write(f"**{c}**")
            vc = df[c].value_counts().reset_index().rename(columns={"index": c, c: "Count"})
            st.dataframe(vc)

# ────────────────────────────────────────────────────────────────────────────────
# 5) PANEL: DISTRIBUTION & OUTLIER ANALYSIS
# ────────────────────────────────────────────────────────────────────────────────

if show_dist:
    st.header("2. Distribution & Outlier Analysis")

    # 5.1) Numeric distribution & outliers
    col = st.selectbox("Select a numeric column:", numeric_cols)
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    ax_hist.hist(df[col].dropna(), bins=20, edgecolor="black")
    ax_hist.set_title(f"Histogram of {col}")
    ax_hist.set_xlabel(col)
    ax_hist.set_ylabel("Frequency")
    st.pyplot(fig_hist)

    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    st.write(f"**Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}**")
    st.write(f"Bounds: [{lower:.2f}, {upper:.2f}], Outliers: {mask.sum()}")

    if st.checkbox("Show outlier rows"):
        st.dataframe(df.loc[mask].head(10))

    if st.checkbox("Show boxplot for all numeric"):
        fig_bp, ax_bp = plt.subplots(figsize=(10, 4))
        ax_bp.boxplot([df[c].dropna() for c in numeric_cols], labels=numeric_cols, vert=False)
        ax_bp.set_title("Boxplot of All Numeric Columns")
        st.pyplot(fig_bp)

    st.markdown("---")

    # 5.2) Pie chart for categorical distribution (exclude Project ID)
    if pie_cols:
        st.subheader("Categorical Distribution (Pie Chart)")
        cat = st.selectbox("Select a categorical column:", pie_cols)  # only "Type"
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

    # 6.1) Filter by Type
    selected_types = st.multiselect("Filter by Type:", df["Type"].unique())

    filtered_df = df.copy()
    if selected_types:
        filtered_df = filtered_df[filtered_df["Type"].isin(selected_types)]

    # 6.2) Pearson correlation on filtered subset
    corr_matrix = filtered_df[numeric_cols].corr()
    st.write("### Pearson Correlation Matrix")
    st.dataframe(corr_matrix)

    if st.checkbox("Show heatmap"):
        fig_hm, ax_hm = plt.subplots(figsize=(6, 6))
        im = ax_hm.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax_hm.set_xticks(range(len(numeric_cols)))
        ax_hm.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax_hm.set_yticks(range(len(numeric_cols)))
        ax_hm.set_yticklabels(numeric_cols)
        fig_hm.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        st.pyplot(fig_hm)

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
        p_values = compute_pvalues(filtered_df[numeric_cols])
        st.write("### P-Value Matrix")
        st.dataframe(p_values)

    if st.checkbox("Show R² (Correlation²) Matrix"):
        r_squared = corr_matrix ** 2
        st.write("### R² Matrix")
        st.dataframe(r_squared)

# ────────────────────────────────────────────────────────────────────────────────
# 7) PANEL: PAIRWISE SCATTER PLOTS
# ────────────────────────────────────────────────────────────────────────────────

if show_scatter:
    st.header("4. Pairwise Scatter Plots")

    # 7.1) Filter by Type
    sel_types = st.multiselect("Filter by Type:", df["Type"].unique(), key="scatter_type")

    df_sc = df.copy()
    if sel_types:
        df_sc = df_sc[df_sc["Type"].isin(sel_types)]

    # 7.2) Select X‐axis and Y‐axis
    x_col = st.selectbox("X-axis column:", numeric_cols, index=0, key="scatter_x")
    y_col = st.selectbox("Y-axis column:", numeric_cols, index=1, key="scatter_y")

    # 7.3) Highlight a specific Project ID
    available_projects = df_sc["Project ID"].unique().tolist()
    sel_project = st.selectbox("Highlight a specific Project ID (optional):", [""] + available_projects)

    fig_sc, ax_sc = plt.subplots(figsize=(5, 4))
    # Plot all points in gray
    ax_sc.scatter(df_sc[x_col], df_sc[y_col], alpha=0.3, color="gray")

    # Overlay selected project in red
    if sel_project:
        proj_row = df_sc[df_sc["Project ID"] == sel_project]
        if not proj_row.empty:
            ax_sc.scatter(proj_row[x_col], proj_row[y_col], s=100, color="red", label=sel_project)
            ax_sc.legend()

    ax_sc.set_xlabel(x_col)
    ax_sc.set_ylabel(y_col)
    ax_sc.set_title(f"{x_col} vs. {y_col}")
    st.pyplot(fig_sc)

    # 7.4) Compute correlation on filtered subset
    valid_x = df_sc[x_col].dropna()
    valid_y = df_sc[y_col].dropna()
    if len(valid_x) > 1 and len(valid_y) > 1:
        r_val, p_val = pearsonr(valid_x, valid_y)
        st.write(f"Pearson r = {r_val:.2f}, p-value = {p_val:.3e}, R² = {r_val**2:.2f}")
    else:
        st.write("Not enough data to compute correlation for selected filters.")

    if st.checkbox("Show full scatter matrix for selected columns"):
        cols_sel = st.multiselect("Pick ≥2 columns:", numeric_cols, default=numeric_cols[:3], key="scatter_matrix_cols")
        if len(cols_sel) >= 2:
            fig_sm2 = scatter_matrix(df_sc[cols_sel], alpha=0.6, diagonal="hist", figsize=(8, 8))
            st.pyplot(plt.gcf())

# ────────────────────────────────────────────────────────────────────────────────
# 8) PANEL: NORMALIZATION & CLUSTERING
# ────────────────────────────────────────────────────────────────────────────────

if show_norm:
    st.header("5. Normalization & (Optional) Clustering")

    if st.checkbox("Show Z-score normalized data"):
        df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        st.write("### First 5 rows of normalized data")
        st.dataframe(df_norm.head())
        st.download_button(
            label="Download normalized CSV",
            data=df_norm.to_csv(index=False).encode("utf-8"),
            file_name="buildings_data_normalized.csv",
            mime="text/csv"
        )

    if st.checkbox("Run K-Means clustering"):
        df_norm = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
        k = st.slider("Number of clusters (k):", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=42).fit(df_norm)
        df["Cluster"] = kmeans.labels_
        st.write("Cluster labels added to the DataFrame.")

        pca = PCA(n_components=2)
        pcs = pca.fit_transform(df_norm)
        fig_pca2, ax_pca2 = plt.subplots(figsize=(5, 4))
        ax_pca2.scatter(pcs[:, 0], pcs[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.7)
        ax_pca2.set_xlabel("PC 1")
        ax_pca2.set_ylabel("PC 2")
        ax_pca2.set_title("PCA Projection by K-Means Cluster")
        st.pyplot(fig_pca2)

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

    top10 = df.sort_values(by=metric, ascending=ascending).head(10)
    st.write(f"Top 10 by **{metric}** ({order}):")
    st.dataframe(top10.reset_index(drop=True))

    st.subheader(f"Top 10 Project IDs vs. {metric}")
    fig_t102, ax_t102 = plt.subplots(figsize=(6, 4))
    ax_t102.barh(top10["Project ID"].astype(str), top10[metric], color="skyblue")
    ax_t102.set_xlabel(metric)
    ax_t102.set_ylabel("Project ID")
    ax_t102.invert_yaxis()
    ax_t102.set_title(f"Top 10 by {metric}")
    st.pyplot(fig_t102)

# ────────────────────────────────────────────────────────────────────────────────
# 10) PANEL: DOWNLOAD REPORTS
# ────────────────────────────────────────────────────────────────────────────────

if show_dl:
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

    df_norm2 = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
    st.download_button("Download buildings_data_normalized.csv", df_norm2.to_csv(index=False).encode("utf-8"), "buildings_data_normalized.csv", "text/csv")

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
    insight_md += "- Use histograms/boxplots to decide if medians or transforms are needed.  \n"
    insight_md += "- Review strong correlations before modeling.  \n"
    insight_md += "- Consider segmenting by Type for deeper benchmarks.  \n"
    insight_md += "- Use normalized data (`buildings_data_normalized.csv`) for clustering/regression.  \n"

    st.markdown(insight_md)
