import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MERGED_DIR = "data_handling/merged"
ANOMALY_DIR = "data_handling/merged_anomaly_eliminated_z"
COMPARE_SAVE_DIR = "analysis/compare_merged_vs_anomaly"
os.makedirs(COMPARE_SAVE_DIR, exist_ok=True)

# 비교할 파일명 지정
FILENAME = "B0047.csv"  # 필요시 변경

FEATURE_COLS = [
    'Voltage_measured', 'Current_measured', 'Temperature_measured',
    'Current_load', 'Voltage_load', 'Time'
]

# 데이터 로드
merged_path = os.path.join(MERGED_DIR, FILENAME)
anomaly_path = os.path.join(ANOMALY_DIR, FILENAME)

df_orig = pd.read_csv(merged_path)
df_anom = pd.read_csv(anomaly_path)

# 시각화
plt.figure(figsize=(15, 10))
for i, col in enumerate(FEATURE_COLS):
    plt.subplot(2, 3, i+1)
    plt.plot(df_orig[col], label='Original', color='tab:blue', alpha=0.7)
    plt.plot(df_anom[col], label='Anomaly Eliminated', color='tab:orange', alpha=0.7)
    # z-score 이상치 기준 계산
    series = df_orig[col]
    zscore = (series - series.mean()) / series.std(ddof=0)
    outlier_mask = np.abs(zscore) >= 10
    # 이상치 위치에 빨간 점 표시
    plt.scatter(df_orig.index[outlier_mask], series[outlier_mask], color='red', s=10, label='Outlier' if i==0 else None, zorder=5)
    # 이상치의 cycle_idx 추출 및 출력
    if 'cycle_idx' in df_orig.columns:
        outlier_cycles = df_orig['cycle_idx'][outlier_mask].values
        print(f"{col} 이상치 row의 cycle_idx: {outlier_cycles}")
    plt.title(col)
    if i == 0:
        plt.legend()
    plt.grid(True)
plt.suptitle(f'Original vs Anomaly Eliminated: {FILENAME}\n(빨간 점: z-score≥3 이상치)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(COMPARE_SAVE_DIR, f'{os.path.splitext(FILENAME)[0]}_compare_with_outlier_zscore.png')
plt.savefig(fig_path, dpi=200)
plt.show()
print(f"비교 그래프(이상치 포함, z-score 기준) 저장: {fig_path}")

# Voltage_measured 이상치 row의 cycle_idx: []
# Current_measured 이상치 row의 cycle_idx: []
# Temperature_measured 이상치 row의 cycle_idx: []
# Current_load 이상치 row의 cycle_idx: [66 66 66]
# Voltage_load 이상치 row의 cycle_idx: [66 66]
# Time 이상치 row의 cycle_idx: []