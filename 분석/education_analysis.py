# project_main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# 1. 데이터 로드 및 전처리
# -----------------------------
def load_and_preprocess(file_path):
    """
    CSV 파일을 불러와 전처리하는 함수.
    결측치는 0으로 채움.
    """
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    return df

# 예시: 서울 교육 데이터
education_data = load_and_preprocess("/path/to/영역2_가중치.csv")

# -----------------------------
# 2. 지표 계산 예시 함수
# -----------------------------
def calculate_ratios(df):
    """
    학생-교사 비율, 학급당 학생 수 등 주요 지표 계산
    """
    df['student_teacher_ratio'] = df['students'] / df['teachers']
    df['class_size'] = df['students'] / df['classes']
    return df

education_data = calculate_ratios(education_data)

# -----------------------------
# 3. 시각화 함수
# -----------------------------
def plot_trends(df, col, title):
    """
    연도별 지표 추세 시각화
    """
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x='year', y=col, hue='district')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_trends(education_data, 'student_teacher_ratio', '학생-교사 비율 연도별 추세')
