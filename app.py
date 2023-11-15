# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

# K-means 모델 학습 및 예측 함수
def kmeans_clustering(df, cols_to_train, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df[cols_to_train])

    # 클러스터 할당 및 결과 컬럼 추가
    df['cluster_kmeans'] = model.predict(df[cols_to_train])

    return df, model

# DBSCAN 모델 학습 및 예측 함수
def dbscan_clustering(df, cols_to_train, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster_dbscan'] = model.fit_predict(df[cols_to_train])

    return df, model

# 이상 탐지 결과 시각화 함수
def visualize_outliers(df, x_col, y_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col, palette='viridis', s=60)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.show()  # streamlit에서는 st.pyplot()를 사용하지 않고 plt.show()로 그림을 보여줍니다.

# Streamlit 앱 함수
def main():
    st.title('이상 탐지 결과 시각화')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # 클러스터링을 위한 feature 선택
        cols_to_train = [
            'method_post', 'status_404', 'path_same',
            'ua_cnt', 'has_payload', 'bytes_avg', 'bytes_std'
        ]

        # K-means 클러스터링 수행
        df_entity, kmeans_model = kmeans_clustering(df_entity, cols_to_train)

        # K-means 클러스터 시각화
        visualize_outliers(df_entity, 'method_post', 'status_404', 'cluster_kmeans', 'K-means 클러스터링 결과')

        # DBSCAN 클러스터링 수행
        df_entity, dbscan_model = dbscan_clustering(df_entity, cols_to_train)

        # DBSCAN 클러스터 시각화
        visualize_outliers(df_entity, 'method_post', 'status_404', 'cluster_dbscan', 'DBSCAN 클러스터링 결과')

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

if __name__ == '__main__':
    main()
