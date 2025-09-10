# ----------------------------------------
# CRITIC 가중치 + PCA 기반 진학률 분석
# Google Colab 환경
# ----------------------------------------

# 📦 라이브러리 설치 (Colab 전용)
!pip install pandas numpy scikit-learn matplotlib seaborn

# 📚 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 데이터 불러오기
# ----------------------------------------
file_path = '/content/drive/MyDrive/서울교육 데이터 분석/1차 정리본/정규화/고등학교별_진학률_정규화.csv'
df = pd.read_csv(file_path)

print("✅ 데이터 미리보기")
print(df.head())
print(df.columns)

# ----------------------------------------
# CRITIC 가중치 계산 함수
# ----------------------------------------
def calculate_critic_weights(data):
    """CRITIC 가중치를 계산하는 함수"""
    std_dev = np.std(data, axis=0)
    corr_matrix = np.corrcoef(data, rowvar=False)
    corr_sum = np.sum(1 - np.abs(corr_matrix), axis=0)   # 상관성이 낮을수록 가중치↑
    critic_scores = std_dev * corr_sum
    total = np.sum(critic_scores)

    if total == 0 or np.isnan(total):
        return np.ones_like(critic_scores) / len(critic_scores)
    else:
        return critic_scores / total

# ----------------------------------------
# 분석 대상 설정
# ----------------------------------------
year_col = '연도'
region_col = '학군'
school_col = '고등학교별(2)'
features = ['진학률']

results = []

# ----------------------------------------
# 학군별 분석 진행
# ----------------------------------------
for hakgun, group in df.groupby(region_col):
    X = group[features].values

    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # CRITIC 가중치 계산
    critic_weights = calculate_critic_weights(X_scaled)

    # PCA 적용
    pca = PCA()
    pca_scores = pca.fit_transform(X_scaled)

    # CRITIC 가중치 적용
    weighted_scores = pca_scores * critic_weights

    # 결과 저장
    group_result = pd.DataFrame(
        weighted_scores,
        columns=[f'{hakgun}_PC{i+1}' for i in range(pca_scores.shape[1])]
    )
    group_result[region_col] = hakgun
    group_result[year_col] = group[year_col].values
    group_result[school_col] = group[school_col].values

    results.append(group_result)

# ----------------------------------------
# 최종 결과 병합 + 저장
# ----------------------------------------
final_df = pd.concat(results, ignore_index=True)

save_path = '/content/drive/MyDrive/서울교육 데이터 분석/1차 정리본/영역3_진학률.csv'
final_df.to_csv(save_path, index=False, encoding='utf-8-sig')

print(final_df.head())
