# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

st.set_page_config(page_title="Quick EDA & Insights", layout="wide")

st.title("Quick EDA & Auto-Insights")
st.markdown("Upload a CSV / Excel file, inspect data, plot variables and get automated insights & recommendations.")

@st.cache_data
def load_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded_file)
        else:
            return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None

def summary_stats(df):
    desc = df.describe(include='all').T
    # include counts of unique and top values for non-numeric
    desc['n_unique'] = df.nunique()
    if 'top' in desc.columns:
        pass
    return desc

def missing_summary(df):
    miss = df.isnull().sum().rename("missing_count")
    miss = miss.to_frame().assign(
        missing_pct = lambda d: (d['missing_count'] / len(df) * 100).round(2)
    ).sort_values("missing_count", ascending=False)
    return miss

def correlation_insights(df_numeric, threshold=0.7):
    corr = df_numeric.corr()
    strong_pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                strong_pairs.append((cols[i], cols[j], float(val)))
    strong_pairs_sorted = sorted(strong_pairs, key=lambda x: -abs(x[2]))
    return corr, strong_pairs_sorted

def outlier_iqr(df, numeric_cols):
    outlier_info = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outliers = df[(df[col] < low) | (df[col] > high)][col]
        outlier_info[col] = {
            "n_outliers": int(outliers.shape[0]),
            "pct_outliers": round(100 * outliers.shape[0] / max(1, df.shape[0]), 2),
            "low_threshold": low,
            "high_threshold": high
        }
    return outlier_info

def quick_feature_importance(df, target_col, task='auto'):
    # simple encode and model for feature importance
    df2 = df.copy()
    X = df2.drop(columns=[target_col])
    y = df2[target_col].copy()
    # encode categoricals
    for c in X.select_dtypes(include=['object','category']).columns:
        X[c] = X[c].fillna("NA").astype(str)
        X[c] = LabelEncoder().fit_transform(X[c])
    # encode target if needed
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.fillna("NA").astype(str))
        task = 'classification'
    else:
        y = y.fillna(y.median())
        task = 'regression'
    # fill remaining nans
    X = X.fillna(X.median(numeric_only=True))
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if task == 'classification':
            m = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            m = RandomForestRegressor(n_estimators=100, random_state=42)
        m.fit(X_train, y_train)
        importances = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importances
    except Exception as e:
        return None

# Sidebar: Upload
st.sidebar.header("Upload data")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv','xls','xlsx'])
sample_data = st.sidebar.selectbox("Or try sample dataset", ["(none)","tips (seaborn sample)","iris (sklearn)"])
if uploaded is None and sample_data != "(none)":
    if sample_data == "tips (seaborn sample)":
        import seaborn as sns
        df = sns.load_dataset("tips")
    else:
        from sklearn.datasets import load_iris
        ir = load_iris(as_frame=True)
        df = ir.frame
else:
    df = None

if uploaded:
    df = load_file(uploaded)

if df is None:
    st.info("Upload a file to start or choose a sample dataset from the sidebar.")
    st.stop()

# Basic info
st.subheader("Data preview & basic info")
c1, c2, c3 = st.columns([3,1,1])
with c1:
    st.dataframe(df.head(100))

with c2:
    st.markdown(f"**Rows:** {df.shape[0]}")
    st.markdown(f"**Columns:** {df.shape[1]}")
    st.markdown(f"**Memory usage:** {round(df.memory_usage(deep=True).sum()/1024**2, 2)} MB")
with c3:
    st.markdown("**Duplicate rows:**")
    st.markdown(f"{df.duplicated().sum()}")

# Data types and summary
st.subheader("Types & Summary")
st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)

st.markdown("**Descriptive summary (numeric & object mixed)**")
st.dataframe(summary_stats(df).fillna("").astype(str).head(50))

# Missing values
st.subheader("Missing values")
miss = missing_summary(df)
st.dataframe(miss)
st.markdown("Columns with > 20% missing:")
st.write(miss[miss['missing_pct'] > 20])

# Numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

# Correlation & heatmap
st.subheader("Correlation (numeric columns)")
if len(numeric_cols) >= 2:
    corr, strong_pairs = correlation_insights(df[numeric_cols], threshold=0.7)
    fig = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    if strong_pairs:
        st.markdown("**Strong correlations (abs >= 0.7):**")
        for a,b,v in strong_pairs[:10]:
            st.write(f"{a} ↔ {b} = {v:.3f}")
    else:
        st.write("No strong correlations found with threshold 0.7.")
else:
    st.write("Not enough numeric columns for correlation.")

# Univariate analysis
st.subheader("Univariate analysis")
col = st.selectbox("Select column to inspect", options=df.columns, index=0)
if pd.api.types.is_numeric_dtype(df[col]):
    st.write(df[col].describe().to_frame())
    fig = px.histogram(df, x=col, nbins=40, marginal="box")
    st.plotly_chart(fig, use_container_width=True)
else:
    vc = df[col].value_counts().reset_index()
    vc.columns = [col, "count"]
    st.dataframe(vc.head(50))
    if vc.shape[0] <= 50:
        fig = px.bar(vc, x=col, y="count")
        st.plotly_chart(fig, use_container_width=True)

# Bivariate
st.subheader("Bivariate analysis")
x_col = st.selectbox("X (for scatter / box)", options=numeric_cols+cat_cols, index=0, key="xcol")
y_col = st.selectbox("Y (numeric preferred)", options=numeric_cols+cat_cols, index=1, key="ycol")
if x_col and y_col:
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.scatter(df, x=x_col, y=y_col, marginal_x="histogram", marginal_y="histogram")
        st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_numeric_dtype(df[y_col]) and not pd.api.types.is_numeric_dtype(df[x_col]):
        fig = px.box(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_numeric_dtype(df[x_col]) and not pd.api.types.is_numeric_dtype(df[y_col]):
        fig = px.box(df, x=y_col, y=x_col)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # both categorical
        ct = pd.crosstab(df[x_col].fillna("NA"), df[y_col].fillna("NA"))
        st.dataframe(ct)
        fig = px.imshow(ct.values, x=ct.columns, y=ct.index, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

# Outliers via IQR
st.subheader("Outlier detection (IQR method)")
if len(numeric_cols) > 0:
    out_info = outlier_iqr(df, numeric_cols)
    out_df = pd.DataFrame.from_dict(out_info, orient='index')
    st.dataframe(out_df.sort_values('n_outliers', ascending=False).head(50))
else:
    st.write("No numeric columns.")

# Automated insights
st.subheader("Automated textual insights & recommendations")
insights = []

# size & shape
insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# missing
total_missing = df.isnull().sum().sum()
if total_missing == 0:
    insights.append("No missing values found.")
else:
    insights.append(f"Total missing cells: {int(total_missing)} ({round(100*total_missing/(df.shape[0]*df.shape[1]),2)}% of all cells). Columns with most missing: " +
                    ", ".join(miss.head(5).index.astype(str)))

# duplicates
dups = int(df.duplicated().sum())
if dups > 0:
    insights.append(f"There are {dups} duplicate rows. Consider dropping duplicates if they are unimportant.")

# constant columns
const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
if const_cols:
    insights.append("Constant or nearly-constant columns: " + ", ".join(const_cols))

# numeric skewness and top means
if len(numeric_cols) > 0:
    skew = df[numeric_cols].skew().abs().sort_values(ascending=False)
    top_skew = skew.head(5)
    insights.append("Top skewed numeric columns: " + ", ".join([f"{i} ({skew[i]:.2f})" for i in top_skew.index]))
    top_means = df[numeric_cols].mean().sort_values(ascending=False).head(5)
    insights.append("Top means (numeric): " + ", ".join([f"{i} ({top_means[i]:.2f})" for i in top_means.index]))

# correlations
if len(numeric_cols) >= 2:
    _, strong_pairs = correlation_insights(df[numeric_cols], threshold=0.7)
    if strong_pairs:
        insights.append("Strong correlations found: " + "; ".join([f"{a}~{b}={v:.2f}" for a,b,v in strong_pairs[:5]]))
    else:
        insights.append("No very strong correlations (abs >= 0.7) among numeric columns.")

# outliers summary
if len(numeric_cols) > 0:
    out_df2 = pd.DataFrame.from_dict(out_info, orient='index')
    maybe_many_outliers = out_df2[out_df2['pct_outliers'] > 5]
    if not maybe_many_outliers.empty:
        insights.append("Columns with >5% outliers (IQR): " + ", ".join(maybe_many_outliers.index.tolist()))

# duplicates, cardinality
high_card_cols = [c for c in df.columns if df[c].nunique() > 0.9 * len(df)]
if high_card_cols:
    insights.append("High-cardinality columns: " + ", ".join(high_card_cols[:10]))

# display insights
for i, s in enumerate(insights):
    st.write(f"- {s}")

# Quick model (optional)
st.subheader("Quick feature importance (optional)")
with st.expander("Train quick model to get feature importances"):
    target = st.selectbox("Select target column (for supervised importance)", options=[None]+list(df.columns), index=0, key="targetcol")
    if target:
        st.info("This will run a quick RandomForest on available columns (encodes categoricals).")
        if st.button("Run quick feature importance"):
            st.write("Training... (may take a few seconds)")
            importances = quick_feature_importance(df, target)
            if importances is None:
                st.error("Could not compute feature importances (maybe too many non-numeric/unencodable columns).")
            else:
                st.dataframe(importances.reset_index().rename(columns={"index":"feature", 0:"importance"}).head(50))
                fig = px.bar(importances.reset_index().rename(columns={"index":"feature", 0:"importance"}), x='feature', y=0)
                st.plotly_chart(fig, use_container_width=True)

# Download cleaned sample (drop high-missing or duplicates) - small helper
st.subheader("Quick cleanup & download")
with st.expander("Create a cleaned sample (drop columns > 50% missing, drop duplicates)"):
    pct = st.slider("Drop columns with missing % >= ", 0, 100, 50)
    if st.button("Create cleaned CSV"):
        cols_to_drop = miss[miss['missing_pct'] >= pct].index.tolist()
        df_clean = df.drop(columns=cols_to_drop).drop_duplicates()
        towrite = BytesIO()
        df_clean.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button("Download cleaned CSV", data=towrite, file_name="cleaned_data.csv", mime="text/csv")

st.markdown("---")
st.caption("App created to speed up first-pass EDA. For deeper analysis consider domain-specific checks, time-series specific tools, and dedicated profiling packages.")
