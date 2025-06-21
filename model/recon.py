import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from collections import defaultdict
from models.transceiver import DeepSC
import matplotlib.pyplot as plt

def reconstruct_battery_series():
    # 데이터 및 메타 정보 로드
    test_data = torch.load('model/preprocessed_data/test_data.pt')
    test_tensor = test_data.tensors[0]
    scaler = joblib.load('model/preprocessed_data/scaler.pkl')
    with open('model/preprocessed_data/window_meta.pkl', 'rb') as f:
        window_meta = pickle.load(f)
    train_data = torch.load('model/preprocessed_data/train_data.pt')
    train_len = len(train_data.tensors[0])

    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]
    window_size = test_tensor.shape[1]
    model = DeepSC(
        num_layers=4,
        input_dim=input_dim,
        max_len=window_size,
        d_model=128,
        num_heads=8,
        dff=512,
        dropout=0.1
    ).to(device)
    checkpoint_path = 'checkpoints/250619/deepsc_battery_epoch40.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # 1. 배터리별로 시계열 길이 추정 (원본 csv에서)
    battery_files = sorted(set([meta['file'] for meta in window_meta]))
    battery_lengths = {}
    for fname in battery_files:
        df = pd.read_csv(os.path.join('data_handling/merged', fname))
        battery_lengths[fname] = len(df)

    # 2. 배터리별로 빈 시계열 배열 준비 (복원값, 카운트)
    reconstructed = {fname: np.zeros((battery_lengths[fname], input_dim)) for fname in battery_files}
    counts = {fname: np.zeros(battery_lengths[fname]) for fname in battery_files}

    # 3. 각 window 복원 및 배터리별 시계열에 합치기 (디버깅 정보 포함)
    with torch.no_grad():
        for i in tqdm(range(test_tensor.shape[0]), desc="Reconstructing"):
            input_data = test_tensor[i].unsqueeze(0).to(device)
            output = model(input_data)
            output_original = scaler.inverse_transform(output.squeeze(0).cpu().numpy())  # (window, feature)
            meta = window_meta[train_len + i]  # test set은 train 다음부터 시작
            fname = meta['file']
            start = meta['start']
            end = start + window_size
            print(f"복원 window {i}: {fname} {start}-{end}, output mean={output_original.mean():.4f}")
            reconstructed[fname][start:end] += output_original
            counts[fname][start:end] += 1

    # 4. 겹치는 부분 평균내기
    for fname in battery_files:
        mask = counts[fname] > 0
        reconstructed[fname][mask] /= counts[fname][mask][:, None]

     # 5. 배터리별로 csv 저장 및 비교 시각화
    save_dir = 'reconstructed'
    os.makedirs(save_dir, exist_ok=True)
    for fname in battery_files:
        base = os.path.splitext(fname)[0]
        df_recon = pd.DataFrame(reconstructed[fname], columns=feature_cols)
        csv_path = os.path.join(save_dir, f'{base}_reconstructed.csv')
        df_recon.to_csv(csv_path, index=False)
        print(f"복원된 전체 시계열 저장: {csv_path}")

        # 비교 시각화 (test set에 포함된 배터리만)
        if np.any(counts[fname] > 0):
            df_orig = pd.read_csv(os.path.join('data_handling/merged', fname))
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                plt.plot(df_orig[col], label='Original', alpha=0.7)
                plt.plot(df_recon[col], label='Reconstructed', alpha=0.7)
                plt.title(col)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'Comparison: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_path = os.path.join(save_dir, f'{base}_compare.png')
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"비교 그래프 저장: {fig_path}")

    # 6. 카운트 배열 확인
    for fname in battery_files:
        print(f"{fname} counts unique: {np.unique(counts[fname])}")

if __name__ == "__main__":
    reconstruct_battery_series()
