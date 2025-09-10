import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data2 = pd.read_csv("data/영역2_종합정규화+복지지수.csv")
data3 = pd.read_csv("data/영역3_종합정규화+복지지수.csv")

df1 = data2.drop(columns=["Unnamed: 0", "교육복지지수"])
df2 = data3.drop(columns=["Unnamed: 0", "교육복지지수"])
merged = pd.merge(df1, df2, on="학군").rename(columns=lambda x: x.replace("(정규화)", "").strip()).set_index("학군")

def plot_heatmap(cmap, filename):
    plt.figure(figsize=(12, 6))
    sns.heatmap(merged, annot=True, cmap=cmap, cbar=True, fmt=".2f")
    plt.title(f"학군별 교육복지 특성 정규화 점수 히트맵 ({cmap})")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"시각화자료/{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_heatmap("YlGnBu", "히트맵_푸른버전")
plot_heatmap("Reds", "히트맵_붉은버전")
plot_heatmap("Blues", "히트맵_파란버전")
plot_heatmap("Greens", "히트맵_초록버전")
