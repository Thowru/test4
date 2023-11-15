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

# Streamlit app
def main():
    st.title("Anomaly Detection with Streamlit")

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

        # Visualize anomalies
        st.subheader("Visualize Anomalies")
        anomaly_hosts_kmeans = df[df['cluster_kmeans'] == 0].index
        anomaly_hosts_dbscan = df[df['cluster_dbscan'] != 0].index
        anomaly_hosts_intersection = anomaly_hosts_kmeans.intersection(anomaly_hosts_dbscan)

        plt.figure(figsize=(10, 6))
        plt.scatter(df['method_post'], df['status_404'], c='blue', label='Normal')
        plt.scatter(df.loc[anomaly_hosts_intersection, 'method_post'], df.loc[anomaly_hosts_intersection, 'status_404'], c='red', label='Anomalies')
        plt.xlabel('method_post')
        plt.ylabel('status_404')
        plt.legend()
        plt.title('Anomalies Visualization')
        st.pyplot()

        # Display anomaly entities
        st.subheader("Anomaly Entities")
        anomaly_entities_df = df.loc[anomaly_hosts_intersection, ['method_cnt', 'method_post', 'protocol_1_0', 'status_major', 'status_404', 'status_499', 'status_cnt', 'path_same', 'path_xmlrpc', 'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std']]
        st.dataframe(anomaly_entities_df)

if __name__ == "__main__":
    main()
