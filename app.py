import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

st.set_page_config(page_title="Student Clustering", layout="wide")
st.title("🎓 ClusterScope")
st.subheader("From DATA to DECISIONS: uncover how students really learn!")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ---------------- DATA PREVIEW ----------------
    st.subheader("Dataset Preview")
    st.write(df.head())

    # ---------------- DATA ANALYSIS ----------------
    st.subheader("📊 Data Analysis")

    st.write("### Summary Statistics")
    st.write(df.describe())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:
        st.write("### Correlation Matrix")
        st.write(numeric_df.corr())

        st.write("### Distributions")
        for col in numeric_df.columns[:3]:
            st.write(f"Distribution of {col}")
            st.bar_chart(numeric_df[col].value_counts().sort_index())

    # ---------------- FEATURE SELECTION ----------------
    features = st.multiselect(
        "Select Features",
        df.columns,
        default=list(numeric_df.columns[:2])
    )

    if len(features) < 2:
        st.warning("Select at least 2 features")
        st.stop()

    df_selected = df[features].select_dtypes(include="number")

    if df_selected.shape[1] < 2:
        st.error("Selected features must be numeric.")
        st.stop()

    # ---------------- PREPROCESSING ----------------
    df_selected = df_selected.fillna(df_selected.mean())

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected),
        columns=df_selected.columns
    )

    # ---------------- ELBOW METHOD ----------------
    st.subheader("Elbow Method")

    inertia = []
    k_range = range(2, 10)

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(df_scaled)
        inertia.append(model.inertia_)

    fig1, ax1 = plt.subplots(figsize= (6, 4))
    ax1.plot(k_range, inertia, marker='o')
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method for Optimal K")
    st.pyplot(fig1)

    st.write("""
    **Interpretation:**
    - X-axis: Number of clusters (K)
    - Y-axis: Inertia (cluster compactness)
    - Choose K where the curve bends (elbow point)
    """)

    # Knee detection
    kl = KneeLocator(k_range, inertia, curve="convex", direction="decreasing")
    k = kl.elbow if kl.elbow else 3

    st.write(f"Optimal K: {k}")

    # ---------------- CLUSTERING ----------------
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    clusters = model.fit_predict(df_scaled)

    df_result = df_selected.copy()
    df_result["Cluster"] = clusters

    # ---------------- EVALUATION ----------------
    if k > 1:
        score = silhouette_score(df_scaled, clusters)
        st.write(f"Silhouette Score: {round(score, 3)}")

    # ---------------- OUTPUT ----------------
    st.subheader("Clustered Data")
    st.write(df_result.head())

    # ---------------- VISUALIZATION ----------------
    st.subheader("Cluster Visualization")

    numeric_cols = df_result.select_dtypes(include="number").columns

    x_axis = st.selectbox("X-axis", numeric_cols)
    y_axis = st.selectbox("Y-axis", numeric_cols, index=1)

    fig2, ax2 = plt.subplots(figsize= (6, 4))
    scatter = ax2.scatter(
        df_result[x_axis],
        df_result[y_axis],
        c=df_result["Cluster"]
    )

    ax2.set_xlabel(x_axis)
    ax2.set_ylabel(y_axis)
    ax2.set_title("Cluster Distribution")

    st.pyplot(fig2)

    st.write(f"""
    **Interpretation:**
    - X-axis: {x_axis}
    - Y-axis: {y_axis}
    - Each color represents a cluster
    - Points in same color = similar students
    """)

    # ---------------- INSIGHTS ----------------
    st.subheader("Cluster Insights")
    st.write(df_result.groupby("Cluster").mean())

    st.subheader("Cluster Interpretation")

    cluster_means = df_result.groupby("Cluster").mean()

    for i in cluster_means.index:
        st.write(f"### Cluster {i}")

        row = cluster_means.loc[i]

        insights = []

        for col in row.index:
            if row[col] > df_selected[col].mean():
                insights.append(f"High {col}")
            else:
                insights.append(f"Low {col}")

        st.write(", ".join(insights))

else:
    st.info("Upload a dataset to begin")
