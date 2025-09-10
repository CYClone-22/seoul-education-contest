import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ====================================
# [영역 2] 학교 환경 지표
# ====================================

# 1. 연도별 평균 계산 함수
def calculate_mean(df, value_cols):
    df_mean = df.copy()
    df_mean['평균'] = df[value_cols].mean(axis=1)
    return df_mean[['학군', '평균']]

# 교원당 학생수 평균
교원당학생수_mean = calculate_mean(data1, [col for col in data1.columns if '교원당학생수' in col])

# 학급당 학생수 평균
학급당학생수_mean = calculate_mean(data2, [col for col in data2.columns if '학급당학생수' in col])

# 사립학교 비율 (이미 평균 제공됨)
사립학교비율 = data3.copy()

# 데이터 병합
merged = 교원당학생수_mean.merge(학급당학생수_mean, on='학군', suffixes=('_교원당학생수', '_학급당학생수'))
merged = merged.merge(사립학교비율[['학군', '사립학교비율']], on='학군')

# 부정지표 반전 처리 (낮을수록 좋음 → 반전)
merged['교원당학생수_반전'] = -merged['평균_교원당학생수']
merged['학급당학생수_반전'] = -merged['평균_학급당학생수']
merged['사립학교비율_반전'] = -merged['사립학교비율']

# 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(merged[['교원당학생수_반전', '학급당학생수_반전', '사립학교비율_반전']])
scaled_df = pd.DataFrame(scaled, columns=['교원당학생수(정규화)', '학급당학생수(정규화)', '사립학교비율(정규화)'])
final_env = pd.concat([merged[['학군']], scaled_df], axis=1)

# CRITIC 가중치 계산
features = final_env[['교원당학생수(정규화)', '학급당학생수(정규화)', '사립학교비율(정규화)']]
std_devs = features.std()
corr_matrix = features.corr()

info = []
n = len(features.columns)
for i in range(n):
    corr_sum = corr_matrix.iloc[i, :].sum() - 1
    info_i = std_devs[i] * (1 - corr_sum / (n - 1))
    info.append(info_i)

critic_weights_env = np.array(info) / sum(info)

weights_df_env = pd.DataFrame({
    '지표': features.columns,
    'CRITIC_가중치': critic_weights_env
})

# 교육복지지수 산출
final_env['교육복지지수'] = features.values @ critic_weights_env


# ====================================
# [영역 3] 지역사회 및 복지 인프라
# ====================================

# 교육예산 평균 (2010~2015)
data5['교육예산_평균'] = data5.iloc[:, 1:7].mean(axis=1)

# 공공도서관 평균 (2018~2023)
data6['공공도서관_평균'] = data6.iloc[:, 1:7].mean(axis=1)

# 청소년시설 평균 (2018~2023)
data7['청소년시설_평균'] = data7.iloc[:, 1:7].mean(axis=1)

# 데이터 병합
merged2 = pd.merge(data5[['학군', '교육예산_평균']], data6[['학군', '공공도서관_평균']], on='학군')
merged2 = pd.merge(merged2, data7[['학군', '청소년시설_평균']], on='학군')

# 정규화
scaler = MinMaxScaler()
scaled = scaler.fit_transform(merged2[['교육예산_평균', '공공도서관_평균', '청소년시설_평균']])
scaled_df = pd.DataFrame(scaled, columns=['교육예산(정규화)', '공공도서관(정규화)', '청소년시설(정규화)'])
final_infra = pd.concat([merged2[['학군']], scaled_df], axis=1)

# CRITIC 가중치 계산
norm_data = final_infra[['교육예산(정규화)', '공공도서관(정규화)', '청소년시설(정규화)']]
std = norm_data.std()
corr_matrix = norm_data.corr().values

information = []
for j in range(len(std)):
    c_j = std[j] * sum(1 - corr_matrix[j])
    information.append(c_j)

information = np.array(information)
critic_weights_infra = information / information.sum()

weights_df_infra = pd.DataFrame({
    '지표': ['교육예산', '공공도서관', '청소년시설'],
    'CRITIC_가중치': critic_weights_infra
})

# 교육복지지수 산출
final_infra['교육복지지수'] = (
    final_infra['교육예산(정규화)'] * critic_weights_infra[0] +
    final_infra['공공도서관(정규화)'] * critic_weights_infra[1] +
    final_infra['청소년시설(정규화)'] * critic_weights_infra[2]
)


# ====================================
# 결과 출력
# ====================================

print("=== 영역 2: 학교 환경 ===")
print(weights_df_env)
print(final_env[['학군', '교육복지지수']])

print("\n=== 영역 3: 지역사회 및 복지 인프라 ===")
print(weights_df_infra)
print(final_infra[['학군', '교육복지지수']])
