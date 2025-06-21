import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os

# 데이터 로드
battery_id = "B0054"  # 예시 배터리 ID
df = pd.read_csv(f"./merged/{battery_id}.csv")
measure_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load']

# figure 저장
output_dir = f"./images/{battery_id}"
# os.makedirs(output_dir, exist_ok=True)

# 컬러맵 준비 (녹색 → 노랑 → 빨강)
norm = mcolors.Normalize(vmin=df['cycle_idx'].min(), vmax=df['cycle_idx'].max())
cmap = cm._colormaps['RdYlGn_r']  # 최신 방식

# 컬럼별 그래프
for col in measure_cols:
    fig, ax = plt.subplots(figsize=(12, 5))
    for cycle_idx, group in df.groupby("cycle_idx"):
        color = cmap(norm(cycle_idx))
        ax.plot(group["Time"], group[col], color=color, label=f"Cycle {cycle_idx}", alpha=0.5)

    ax.set_title(f"battery:{battery_id} {col} vs Time (Colored by Cycle)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(col)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # colorbar에 ax 명시
    plt.colorbar(sm, ax=ax, label='Cycle Index')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 저장
    save_path = os.path.join(output_dir, f"{battery_id}_{col}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


# # 각 cycle의 시작/마지막 전압값만 추출
# start_voltage = df.groupby('cycle_idx').apply(lambda x: x.iloc[0]['Voltage_measured'])
# end_voltage = df.groupby('cycle_idx').apply(lambda x: x.iloc[-1]['Voltage_measured'])
# end_curr_measured = df.groupby('cycle_idx').apply(lambda x: x.iloc[-1]['Current_measured'])
# end_temp = df.groupby('cycle_idx').apply(lambda x: x.iloc[-1]['Temperature_measured'])
# end_curr_load = df.groupby('cycle_idx').apply(lambda x: x.iloc[-1]['Current_load'])

# # print(type(voltage_grouped), type(start_voltage))

# # 시각화
# plt.figure(figsize=(10, 5))
# plt.plot(voltage_grouped.index, voltage_grouped.values, marker='o')
# plt.title("Start/End-of-Discharge Voltage vs Cycle Index")
# plt.xlabel("Cycle Index")
# plt.ylabel("Voltage (V)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # 시각화: 온도 변화
# plt.figure(figsize=(10, 5))
# plt.plot(curr_load.index, curr_load.values, marker='o', color='orange')
# plt.title("Temperature vs Cycle Index")
# plt.xlabel("Cycle Index")
# plt.ylabel("Temperature (°C)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()