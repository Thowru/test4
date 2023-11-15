import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle

# Load Data Function
def load_data():
    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])

    # 파일이 업로드되었을 때의 처리
    if uploaded_file is not None:
        # 업로드한 파일을 Pandas DataFrame으로 읽어오기
        df_entity = pd.read_csv(uploaded_file)
        return df_entity
    else:
        return None

# K-means 클러스터링 함수
def kmeans_clustering(df, cols_to_train, n_clusters=2):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols_to_train])

    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df_scaled)

    df['cluster_kmeans'] = model.labels_

    return df, model

# DBSCAN 클러스터링 함수
def dbscan_clustering(df, cols_to_train, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols_to_train])

    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = model.fit_predict(df_scaled)

    return df, model

# 시각화 함수
def visualize_clusters_2d(df, x_col, y_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col, palette='viridis', s=60)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.colorbar(label='클러스터')
    st.pyplot()

# Streamlit 앱 함수
def main():
    st.title('이상 탐지 결과 시각화')

    # 데이터 로드
    df_entity = load_data()

    if df_entity is not None:
        # 클러스터링을 위한 feature 선택
        cols_to_train = ['method_cnt', 'method_post', 'protocol_1_0', 'status_major', 'status_404', 'status_499', 'status_cnt', 'path_same', 'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std']

        # K-means 클러스터링 수행
        df_entity, kmeans_model = kmeans_clustering(df_entity, cols_to_train)

        # K-means 클러스터 시각화
        visualize_clusters_2d(df_entity, 'method_post', 'status_404', 'cluster_kmeans', 'K-means 클러스터링 결과')

        # DBSCAN 클러스터링 수행
        df_entity, dbscan_model = dbscan_clustering(df_entity, cols_to_train)

        # DBSCAN 클러스터 시각화
        visualize_clusters_2d(df_entity, 'method_post', 'status_404', 'cluster_dbscan', 'DBSCAN 클러스터링 결과')

        # 클러스터별 데이터 포인트 수 출력
        st.write("K-means 클러스터의 데이터 포인트 수:")
        st.write(df_entity['cluster_kmeans'].value_counts())

        st.write("DBSCAN 클러스터의 데이터 포인트 수:")
        st.write(df_entity['cluster_dbscan'].value_counts())

        # K-means 이상 탐지된 Host 출력
        st.write("K-means 이상 탐지된 Host:")
        st.write(df_entity[df_entity['cluster_kmeans'] == -1]['Host'])

        # DBSCAN 이상 탐지된 Host 출력
        st.write("DBSCAN 이상 탐지된 Host:")
        st.write(df_entity[df_entity['cluster_dbscan'] == -1]['Host'])

        # 클러스터링 결과를 기반으로 이상탐지된 클러스터에 속하는 entity 추출 (DBSCAN)
        anomaly_entities_dbscan = df_entity[df_entity['cluster_dbscan'] != 0].index

        # 추출한 entity를 pickle 파일로 저장
        with open('anomaly_entities_dbscan.pkl', 'wb') as f:
            pickle.dump(anomaly_entities_dbscan, f)

        # 클러스터링 결과에서 이상탐지된 클러스터에 속하는 entity 추출 (K-means)
        anomaly_entities_kmeans = df_entity[df_entity['cluster_kmeans'] == 0].index

        # 추출한 entity를 pickle 파일로 저장
        with open('anomaly_entities_kmeans.pkl', 'wb') as f:
            pickle.dump(anomaly_entities_kmeans, f)

if __name__ == '__main__':
    main()
