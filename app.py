import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# K-means 모델 학습 및 예측 함수
def kmeans_clustering(df, cols_to_train, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(df[cols_to_train])

    # 클러스터 할당 및 결과 컬럼 추가
    df['cluster_kmeans'] = model.predict(df[cols_to_train])

    return df, model

# 시각화 함수
def visualize_kmeans_clusters(df, model, cols_to_visualize):
    plt.scatter(df[cols_to_visualize[0]], df[cols_to_visualize[1]], c=df['cluster_kmeans'], s=60)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='^', s=100)
    plt.xlabel(cols_to_visualize[0])
    plt.ylabel(cols_to_visualize[1])
    plt.title("K-means 클러스터링 결과")

    # Streamlit에 Matplotlib 그래프 표시
    st.pyplot()

# Streamlit 앱 함수
def main():
    st.title('K-means 클러스터링 결과 시각화')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # 클러스터링을 위한 feature 선택
        cols_to_train = ['method_post', 'status_404']

        # K-means 클러스터링 수행
        df_entity, kmeans_model = kmeans_clustering(df_entity, cols_to_train)

        # K-means 클러스터 시각화
        visualize_kmeans_clusters(df_entity, kmeans_model, cols_to_train)

        # 클러스터별 데이터 포인트 수 출력
        st.write("각 클러스터의 데이터 포인트 수:")
        st.write(df_entity['cluster_kmeans'].value_counts())

        # 클러스터 0에 속한 Host 출력
        st.write("클러스터 0에 속한 Host:")
        st.write(df_entity[df_entity['cluster_kmeans'] == 0]['Host'])

if __name__ == '__main__':
    main()
