import streamlit as st
import pandas as pd
import numpy as np
import base64

def feature_extract(df):
    df['method_cnt'] = 0.0
    df['method_post'] = 0.0
    df['protocol_1_0'] = False
    df['status_major'] = 0.0
    df['status_404'] = 0.0
    df['status_499'] = False
    df['status_cnt'] = 0.0
    df['path_same'] = 0.0
    df['path_xmlrpc'] = True
    df['ua_cnt'] = 0.0
    df['has_payload'] = False
    df['bytes_avg'] = 0.0
    df['bytes_std'] = 0.0

    cnt = 0
    for entity in df['Host'].unique():
        if cnt % 500 == 0:
            print(cnt)

        group = df[df['Host'] == entity]

        method_cnt = group['Method'].nunique()
        df.loc[df['Host'] == entity, 'method_cnt'] = method_cnt

        method_post_percent = len(group[group['Method'] == 'POST']) / max(1, len(group))
        df.loc[df['Host'] == entity, 'method_post'] = method_post_percent

        use_1_0 = any(group['Protocol'] == 'HTTP/1.0')
        df.loc[df['Host'] == entity, 'protocol_1_0'] = use_1_0

        status_major_percent = len(group[group['Status'].isin(['200', '301', '302'])]) / max(1, len(group))
        df.loc[df['Host'] == entity, 'status_major'] = status_major_percent

        status_404_percent = len(group[group['Status'] == '404']) / max(1, len(group))
        df.loc[df['Host'] == entity, 'status_404'] = status_404_percent

        has_499 = any(group['Status'] == '499')
        df.loc[df['Host'] == entity, 'status_499'] = has_499

        status_cnt = group['Status'].nunique()
        df.loc[df['Host'] == entity, 'status_cnt'] = status_cnt

        top1_path_cnt = group['Path'].value_counts().iloc[0] if not group['Path'].value_counts().empty else 0
        path_same = top1_path_cnt / max(1, len(group))
        df.loc[df['Host'] == entity, 'path_same'] = path_same

        path_xmlrpc = len(group[group['Path'].str.contains('xmlrpc.php') == True]) / max(1, len(group))
        df.loc[df['Host'] == entity, 'path_xmlrpc'] = path_xmlrpc

        ua_cnt = group['UA'].nunique()
        df.loc[df['Host'] == entity, 'ua_cnt'] = ua_cnt

        has_payload = any(group['Payload'] != '-')
        df.loc[df['Host'] == entity, 'has_payload'] = has_payload

        bytes_avg = np.mean(group['Bytes'])
        bytes_std = np.std(group['Bytes'])
        df.loc[df['Host'] == entity, 'bytes_avg'] = bytes_avg
        df.loc[df['Host'] == entity, 'bytes_std'] = bytes_std

        cnt += 1
    return df

def main():
    st.title('로그 데이터 처리 앱')

    # 파일 업로드
    uploaded_csvfile = st.file_uploader("CSV 파일 선택", type="csv")

    if uploaded_csvfile is not None:
        # CSV 파일 읽기
        df_entity = pd.read_csv(uploaded_csvfile)

        # Feature Extraction
        df_entity_processed = feature_extract(df_entity)

        # host 컬럼을 entity로 변경
        df_entity_processed = df_entity_processed.rename(columns={'Host': 'entity'})

        # 불필요한 컬럼 제거
        columns_to_drop = ['Unnamed: 0', 'Timestamp', 'Method', 'Protocol', 'Status', 'Referer', 'Path', 'UA', 'Payload', 'Bytes']
        df_entity_processed = df_entity_processed.drop(columns=columns_to_drop, errors='ignore')

        # 'entity' 컬럼을 맨 앞으로 이동
        columns_order = ['entity'] + [col for col in df_entity_processed.columns if col != 'entity']
        df_entity_processed = df_entity_processed[columns_order]

        # 전처리된 데이터 출력
        st.write("전처리된 데이터:")
        st.write(df_entity_processed)

        # CSV 파일 다운로드 버튼 생성
        csv_file = df_entity_processed.to_csv(index=False).encode()
        b64 = base64.b64encode(csv_file).decode()
        st.button("Download CSV 파일", on_click=lambda: st.markdown(f'<a href="data:file/csv;base64,{b64}" download="preprocessed_data.csv">Download CSV 파일</a>', unsafe_allow_html=True))

if __name__ == '__main__':
    main()
