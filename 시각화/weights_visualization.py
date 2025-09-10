import matplotlib.pyplot as plt

# 영역별 가중치 딕셔너리
area1_weights = {...}  # 그대로 유지
area2_weights = {...}
area3_weights = {...}

plt.rc('font', family='NanumBarunGothic')

def plot_weights(area_weights, area_name, color, filename):
    plt.figure(figsize=(10, 5))
    plt.bar(area_weights.keys(), area_weights.values(), color=color)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('CRITIC 가중치')
    plt.title(f'{area_name} 내 지표별 가중치')
    plt.tight_layout()

    filepath = f"시각화자료/{filename}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

plot_weights(area1_weights, '영역1 - 학생 및 학부모 특성', '#74b9ff', 'area1_weights')
plot_weights(area2_weights, '영역2 - 학교 및 교육환경', '#55efc4', 'area2_weights')
plot_weights(area3_weights, '영역3 - 학생 성과 및 학습지원', '#ffeaa7', 'area3_weights')
