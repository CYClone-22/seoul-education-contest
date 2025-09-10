import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv("data/영역1_교육복지점수.csv")
data2 = pd.read_csv("data/영역2_종합정규화+복지지수.csv")
data3 = pd.read_csv("data/영역3_종합정규화+복지지수.csv")
data_final = pd.read_csv("data/최종_교육복지점수.csv")

merged = data1[['학군', '교육복지지수']].rename(columns={'교육복지지수': '영역1_학생및학부모'})
merged['영역2_학교및환경'] = data2['교육복지지수']
merged['영역3_지역사회복지인프라'] = data3['교육복지지수']
merged['최종교육복지지수'] = data_final['최종_교육복지지수']

bar_width, group_gap = 0.25, 0.4
n = len(merged)
x_base = [i * (3 * bar_width + group_gap) for i in range(n)]
x1, x2, x3 = [x for x in x_base], [x + bar_width for x in x_base], [x + 2 * bar_width for x in x_base]
x_line = [x + 1.5 * bar_width for x in x_base]

plt.figure(figsize=(16, 7))
plt.bar(x1, merged['영역1_학생및학부모'], width=bar_width, label='학생 및 학부모', color='#0984e3')
plt.bar(x2, merged['영역2_학교및환경'], width=bar_width, label='학교 및 교육환경', color='#d63031')
plt.bar(x3, merged['영역3_지역사회복지인프라'], width=bar_width, label='지역사회 및 인프라', color='#fdcb6e')

plt.plot(x_line, merged['최종교육복지지수'], marker='o', color='black', linewidth=2, label='최종 교육복지지수')

plt.xticks([x + 1.5 * bar_width for x in x_base], merged['학군'], rotation=0, fontsize=14)
plt.ylabel('점수')
plt.title('학군별 영역별 교육복지지수 + 최종 지수')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("시각화자료/학군별_영역별+최종_교육복지지수.png", dpi=300)
plt.show()
