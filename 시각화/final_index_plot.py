import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_final = pd.read_csv("data/최종_교육복지점수.csv")
data_final["학군명"] = (data_final.index + 1).astype(str) + " 학군"
data_final = data_final.sort_values(by="최종_교육복지지수", ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(data_final["학군명"], data_final["최종_교육복지지수"], color='#2980b9')

plt.xticks(rotation=0)
plt.yticks(np.arange(0, 1.05, 0.1))
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
plt.ylabel('최종 교육복지지수')
plt.title('학군별 최종 교육복지지수 (내림차순)')
plt.tight_layout()
plt.savefig("시각화자료/학군별_최종_교육복지지수.png", dpi=300)
plt.show()
