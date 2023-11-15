import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# K-means 모델 학습 및 예측 함수
def kmeans_clustering(df, cols_to_train, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df[cols_to_train])

    # 클러스터 할당 및 결과 컬럼 추가
    df['cluster_kmeans'] = model.predict(df[cols_to_train])

    return df, model

# PCA를 사용하여 데이터의 차원을 2로 축소
def apply_pca(df, cols_to_train):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[cols_to_train])
    df['pca_1'] = pca_result[:, 0]
    df['pca_2'] = pca_result[:, 1]
    return df

# t-SNE를 사용하여 데이터의 차원을 2로 축소
def apply_tsne(df, cols_to_train):
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(df[cols_to_train])
    df['tsne_2d_one'] = tsne_results[:, 0]
    df['tsne_2d_two'] = tsne_results[:, 1]
    return df

# 시각화 함수
def visualize_clusters_2d(df, x_col, y_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    
    try:
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=hue_col, palette='viridis', s=60)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.colorbar(label='클러스터')
        st.pyplot()
    except ValueError as e:
        st.error(f"시각화 중 오류 발생: {e}")
        st.write(f"{hue_col} 컬럼에 유효한 값이 있는지 확인하세요.")

# Streamlit 앱 함수
def main():
    st.title('K-means 및 차원 축소 결과 시각화')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # 클러스터링을 위한 feature 선택
        cols_to_train_kmeans = ['method_post', 'status_404']

        # K-means 클러스터링 수행
        df_entity, kmeans_model = kmeans_clustering(df_entity, cols_to_train_kmeans)

        # K-means 클러스터 시각화
        visualize_clusters_2d(df_entity, 'pca_1', 'pca_2', 'cluster_kmeans', 'K-means 클러스터링 결과 (PCA)')

        # PCA를 사용한 차원 축소
        df_entity = apply_pca(df_entity, cols_to_train_kmeans)

        # PCA 시각화
        visualize_clusters_2d(df_entity, 'pca_1', 'pca_2', 'cluster_kmeans', 'PCA를 사용한 차원 축소 결과')

        # t-SNE를 사용한 차원 축소
        df_entity = apply_tsne(df_entity, cols_to_train_kmeans)

        # t-SNE 시각화
        visualize_clusters_2d(df_entity, 'tsne_2d_one', 'tsne_2d_two', 'cluster_kmeans', 't-SNE를 사용한 차원 축소 결과')

if __name__ == '__main__':
    main()
