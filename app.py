import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

# Function to perform K-means clustering
def perform_kmeans(data, n_clusters=2):
    kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
    data['cluster_kmeans'] = kmeans_model.fit_predict(data)
    return data, kmeans_model

# Function to perform DBSCAN clustering
def perform_dbscan(data, eps=0.5, min_samples=2):
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples)
    data['cluster_dbscan'] = dbscan_model.fit_predict(data)
    return data, dbscan_model

# Function to visualize 2D clusters
def visualize_clusters_2d(data, x_col, y_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=x_col, y=y_col, data=data, hue=hue_col, palette='viridis', s=60)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)

    # Extract the mappable from the scatter plot
    handles, labels = scatter.get_legend_handles_labels()
    mappable = handles[0]

    # Add colorbar using the mappable
    plt.colorbar(mappable, label='클러스터')

    st.pyplot()

# Function to visualize 2D PCA clusters
def visualize_pca_clusters(data, x_col, y_col, hue_col, title):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[cols_to_train])

    data['pca_1'] = pca_result[:, 0]
    data['pca_2'] = pca_result[:, 1]

    visualize_clusters_2d(data, 'pca_1', 'pca_2', hue_col, title)

# Function to visualize 2D TSNE clusters
def visualize_tsne_clusters(data, x_col, y_col, hue_col, title):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(data[cols_to_train])

    data['tsne-2d-one'] = tsne_results[:, 0]
    data['tsne-2d-two'] = tsne_results[:, 1]

    visualize_clusters_2d(data, 'tsne-2d-one', 'tsne-2d-two', hue_col, title)

# Streamlit app
def main():
    st.title("Clustering Analysis with Streamlit")

    # File Upload
    st.sidebar.header("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    # K-means parameters
    n_clusters = st.sidebar.slider("Number of K-means clusters", 2, 10, 2)

    # DBSCAN parameters
    eps = st.sidebar.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5)
    min_samples = st.sidebar.slider("DBSCAN Minimum Samples", 1, 10, 2)

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display the DataFrame
        st.subheader("Data Preview")
        st.dataframe(df)

        # K-means clustering
        st.subheader("K-means Clustering")
        df, kmeans_model = perform_kmeans(df, n_clusters)
        st.write("K-means clustering results:")
        st.write("Cluster Centers:", kmeans_model.cluster_centers_)
        st.write("Cluster Labels:", df['cluster_kmeans'].value_counts())

        # DBSCAN clustering
        st.subheader("DBSCAN Clustering")
        df, dbscan_model = perform_dbscan(df, eps, min_samples)
        st.write("DBSCAN clustering results:")
        st.write("Core sample indices:", dbscan_model.core_sample_indices_)
        st.write("Cluster Labels:", df['cluster_dbscan'].value_counts())

        # Visualize 2D clusters
        st.subheader("Visualize Clusters (2D)")
        visualize_clusters_2d(df, 'method_post', 'status_404', 'cluster_kmeans', 'K-means Clustering Results')

        # Visualize 2D PCA clusters
        st.subheader("Visualize PCA Clusters (2D)")
        visualize_pca_clusters(df, 'method_post', 'status_404', 'cluster_kmeans', 'PCA Clustering Results')

        # Visualize 2D TSNE clusters
        st.subheader("Visualize TSNE Clusters (2D)")
        visualize_tsne_clusters(df, 'method_post', 'status_404', 'cluster_kmeans', 'TSNE Clustering Results')

if __name__ == "__main__":
    main()
