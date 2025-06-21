import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
from glob import glob

# 컬러맵 설정
cmap = cm._colormaps['RdYlGn_r']  # 초록 → 노랑 → 빨강

# 측정 컬럼 목록
measure_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']

# README 파일들
readme_files = glob("./extra_infos/README_*.txt")

# 모든 배터리 csv 파일 경로
csv_files = {os.path.splitext(os.path.basename(f))[0]: f for f in glob("./merged/B*.csv")}

for readme_path in readme_files:
    # README 파일명에서 배터리 ID 추출
    basename = os.path.basename(readme_path)
    group_name = os.path.splitext(basename)[0]  # e.g., README_05_06_07_18
    id_part = group_name.replace("README_", "")
    battery_ids = id_part.split("_")

    print(f"README 처리 중: {basename} | 포함 배터리: {battery_ids}")
    
    # 데이터프레임을 병합
    dfs = []
    for bid in battery_ids:
        csv_name = f"B{int(bid):04d}"  # 항상 B00NN 또는 B0NNN 포맷

        if csv_name in csv_files:
            df = pd.read_csv(csv_files[csv_name])
            df["battery_id"] = bid
            dfs.append(df)
        else:
            print(f"CSV 파일 없음: {csv_name}.csv")

    if not dfs:
        print(f"데이터 없음: {basename}")
        continue
    
    df_all = pd.concat(dfs, ignore_index=True)

    # 루프 안에서 group_name 기반으로 디렉토리 생성
    output_dir = f"./images/{group_name}"
    os.makedirs(output_dir, exist_ok=True)

    for col in measure_cols:
        fig, ax = plt.subplots(figsize=(12, 5))

        norm = mcolors.Normalize(vmin=df_all['cycle_idx'].min(), vmax=df_all['cycle_idx'].max())
        colors = plt.cm.tab10.colors
        
        for i, bid in enumerate(battery_ids):
            sub_df = df_all[df_all['battery_id'] == bid]
            for cycle_idx, group in sub_df.groupby("cycle_idx"):
                color_base = colors[i % len(colors)]
                alpha = 0.3 + 0.7 * norm(cycle_idx)
                ax.plot(group["Time"], group[col], color=color_base, alpha=alpha, 
                        label=f"B{bid}-C{cycle_idx}" if cycle_idx == group["cycle_idx"].min() else "")

        ax.set_title(f"Battery Group: {id_part} {col} vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.grid(True)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='best')

        plt.tight_layout(pad=2.0)

        save_path = os.path.join(output_dir, f"{group_name}_{col}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

print("모든 README 기반 배터리 그룹 그래프 저장 완료")
