import numpy as np
import pandas as pd

# ===============================
# 1. 데이터 병합
# ===============================
# 영역별 복지지수 데이터(data1, data2, data3)에서 학군 + 복지지수만 추출 후 병합
merged = pd.DataFrame({
    '학군': data1['학군'],
    '영역1_복지지수': data1['교육복지지수'],
    '영역2_복지지수': data2['교육복지지수'],
    '영역3_복지지수': data3['교육복지지수']
})

# ===============================
# 2. CRITIC 가중치 계산
# ===============================
features = merged[['영역1_복지지수', '영역2_복지지수', '영역3_복지지수']]

# (1) 표준편차
std_devs = features.std()

# (2) 상관계수 행렬
corr_matrix = features.corr()

# (3) 정보량 계산
info = []
n = len(features.columns)
for i in range(n):
    corr_sum = corr_matrix.iloc[i, :].sum() - 1  # 자기 자신 제외
    info_i = std_devs[i] * (1 - corr_sum / (n - 1))
    info.append(info_i)

# (4) 가중치 정규화
critic_weights = np.array(info) / sum(info)

# ===============================
# 3. 결과 저장
# ===============================
# 영역별 가중치
weights_df = pd.DataFrame({
    '영역': ['영역1', '영역2', '영역3'],
    'CRITIC_가중치': critic_weights
})
print(weights_df)

# 최종 교육복지지수
merged['최종_교육복지지수'] = features.values @ critic_weights
print(merged[['학군', '최종_교육복지지수']])
