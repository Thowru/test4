# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import seaborn as sns
import pickle

# Load data
@st.cache  # Use cache to avoid reloading data on each interaction
def load_data():
    # Replace 'your_data.csv' with the actual file path or URL
    df_entity = pd.read_csv('your_data.csv')
    return df_entity

df_entity = load_data()

# Sidebar for user input
st.sidebar.title('Model Configuration')
selected_features = st.sidebar.multiselect('Select Features for Clustering', df_entity.columns)

# K-means model
st.sidebar.subheader('K-means Model')
n_clusters_kmeans = st.sidebar.slider('Number of Clusters (K-means)', 2, 10, 2)
kmeans_model = KMeans(n_clusters=n_clusters_kmeans, random_state=42)
kmeans_model.fit(df_entity[selected_features])
df_entity['cluster_kmeans'] = kmeans_model.predict(df_entity[selected_features])

# DBSCAN model
st.sidebar.subheader('DBSCAN Model')
eps_dbscan = st.sidebar.slider('EPS (DBSCAN)', 0.1, 2.0, 0.5)
min_samples_dbscan = st.sidebar.slider('Min Samples (DBSCAN)', 1, 10, 5)
dbscan_model = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan)
df_entity['cluster_dbscan'] = dbscan_model.fit_predict(df_entity[selected_features])

# Display clustering results
st.title('Clustering Results and Anomaly Detection')

# K-means clustering results
st.subheader('K-means Clustering Results')
st.write(df_entity['cluster_kmeans'].value_counts())

# DBSCAN clustering results
st.subheader('DBSCAN Clustering Results')
st.write(df_entity['cluster_dbscan'].value_counts())

# Visualize Clustering Results
st.subheader('Visualize Clustering Results')

# PCA
st.subheader('PCA Visualization')
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_entity[selected_features])
df_entity['pca_1'] = pca_result[:, 0]
df_entity['pca_2'] = pca_result[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca_1', y='pca_2', hue='cluster_kmeans', data=df_entity, palette='viridis', s=60)
st.pyplot()

# TSNE
st.subheader('t-SNE Visualization')
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500, random_state=42)
tsne_results = tsne.fit_transform(df_entity[selected_features])
df_entity['tsne_2d_one'] = tsne_results[:, 0]
df_entity['tsne_2d_two'] = tsne_results[:, 1]
plt.figure(figsize=(10, 6))
sns.scatterplot(x='tsne_2d_one', y='tsne_2d_two', hue='cluster_kmeans', data=df_entity, palette='viridis', s=60)
st.pyplot()

# Save anomaly entities to pickle files
with open('anomaly_entities_dbscan.pkl', 'wb') as f:
    pickle.dump(df_entity[df_entity['cluster_dbscan'] != 0].index, f)

with open('anomaly_entities_kmeans.pkl', 'wb') as f:
    pickle.dump(df_entity[df_entity['cluster_kmeans'] == 0].index, f)
