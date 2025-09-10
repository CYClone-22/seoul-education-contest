import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
data1 = pd.read_csv("data/영역1_교육복지점수.csv")
data2 = pd.read_csv("data/영역2_종합정규화+복지지수.csv")
data3 = pd.read_csv("data/영역3_종합정규화+복지지수.csv")

# 병합
merged = data1[['학군', '교육복지지수']].rename(columns={'교육복지지수': '영역1_학생및학부모'})
merged['영역2_학교및환경'] = data2['교육복지지수']
merged['영역3_학생성과및지원'] = data3['교육복지지수']

# 시각화
plt.figure(figsize=(12, 6))
x = range(len(merged))
bar_width = 0.25

plt.bar([i - bar_width for i in x], merged['영역1_학생및학부모'], width=bar_width, label='학생 및 학부모 특성', color='#74b9ff')
plt.bar(x, merged['영역2_학교및환경'], width=bar_width, label='학교 및 교육환경', color='#55efc4')
plt.bar([i + bar_width for i in x], merged['영역3_학생성과및지원'], width=bar_width, label='학생 성과 및 학습지원', color='#ffeaa7')

plt.xticks(x, merged['학군'], rotation=45, ha='right')
plt.ylabel('영역별 교육복지지수')
plt.title('학군별 영역별 교육복지지수 비교')
plt.legend()
plt.tight_layout()
plt.savefig("시각화자료/학군별_영역별_교육복지지수.png", dpi=300)
plt.show()
