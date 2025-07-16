import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset
import joblib
import pickle
import matplotlib.pyplot as plt 
import pdb

window_size_arr = [16, 32, 64, 128]
window_size = window_size_arr[3]

def is_valid_csv(fpath, expected_columns):
    try:
        df = pd.read_csv(fpath, nrows=1)
        return all(col in df.columns for col in expected_columns)
    except:
        return False

def load_all_valid_csv_tensors_by_cycle(folder_path, feature_cols, batch_size=8, save_split_path=None, split_ratio=0.8, window_size=128, stride=64, cycle_col='cycle_idx'):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])
    total_files = len(files)
    valid_files = 0
    all_data = []  # 모든 원본 데이터를 먼저 수집
    all_cycles = []  # cycle_idx 정보
    # 'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'

    # 1단계: 모든 데이터 수집
    for fname in tqdm(files, desc="Loading CSV files"):
        fpath = os.path.join(folder_path, fname)
        if not is_valid_csv(fpath, feature_cols + [cycle_col]):
            continue
        try:
            df = pd.read_csv(fpath)
            # 이상치 보정
            # df = correct_outliers_with_interpolation(df, feature_cols)
            df = df[df['Voltage_load'] < 10]
            df = df[df['Current_load'] < 2]
            PREPROCESSED_DIR = "data_handling/merged_preprocessed"
            save_path = os.path.join(PREPROCESSED_DIR, fname)
            df.to_csv(save_path, index=False)

            data = df[feature_cols].values.astype(np.float32)
            cycles = df[cycle_col].values.astype(np.int32)
            all_data.append(data)
            all_cycles.append(cycles)
            valid_files += 1
        except Exception as e:
            print(f"[WARN] Failed to process {fname}: {e}")

    print(f"Valid CSV files loaded: {valid_files} / {total_files}")

    # pdb.set_trace()

    if not all_data:
        print("[ERROR] No valid data found.")
        return

    # 2단계: 전체 데이터를 하나로 합치고 스케일링
    print("Applying scaling to all data...")
    combined_data = np.vstack(all_data)  # 모든 데이터를 세로로 합침
    scaler = MinMaxScaler()
    scaled_combined = scaler.fit_transform(combined_data)

    # pdb.set_trace()

    # 3단계: cycle별로 파일 분리 및 window 생성
    all_windows = []
    window_meta = []  # (파일명, cycle_idx, window_start_index) 저장
    start_idx = 0

    # 전처리된 데이터 저장 폴더 생성
    for fname, data, cycles in zip(files, all_data, all_cycles):
        data_len = len(data)
        end_idx = start_idx + data_len
        scaled_data = scaled_combined[start_idx:end_idx]
        df = pd.DataFrame(scaled_data, columns=feature_cols)
        df[cycle_col] = cycles
        # 전처리된 데이터 저장
        save_path = os.path.join(PREPROCESSED_DIR, fname)
        df.to_csv(save_path, index=False)
        # cycle별로 groupby
        for cycle_id, group in df.groupby(cycle_col):
            group = group.reset_index(drop=True)
            group_len = len(group)
            # progress ratio 계산
            progress_ratio = np.linspace(0, 1, group_len, endpoint=True)
            for win_start in range(0, group_len - window_size + 1, stride):
                window = group.iloc[win_start:win_start + window_size].copy()
                # cycle_idx, progress_ratio feature 추가
                # window['cycle_idx'] = cycle_id
                # window['progress_ratio'] = progress_ratio[win_start:win_start + window_size]
                # # feature 순서: 기존 feature_cols + ['cycle_idx', 'progress_ratio']
                # window_tensor = torch.tensor(window[feature_cols + ['cycle_idx', 'progress_ratio']].values, dtype=torch.float32)
                window_tensor = torch.tensor(window[feature_cols].values, dtype=torch.float32)
                all_windows.append(window_tensor)
                window_meta.append({'file': fname, 'cycle_idx': cycle_id, 'start': win_start})
        start_idx = end_idx

    # 텐서로 변환
    full_tensor = torch.stack(all_windows, dim=0)  # [Total_N, window, D+2]
    
    if save_split_path:
        N = full_tensor.shape[0]
        train_len = int(N * split_ratio)
        train_data = TensorDataset(full_tensor[:train_len])
        test_data = TensorDataset(full_tensor[train_len:])
        torch.save(train_data, os.path.join(save_split_path, 'train_data.pt'))
        torch.save(test_data, os.path.join(save_split_path, 'test_data.pt'))
        print(f"Saved train_data.pt ({train_len} samples), test_data.pt ({N - train_len} samples) to {save_split_path}")

        scaler_path = os.path.join(save_split_path, 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
        # window_meta도 저장
        with open(os.path.join(save_split_path, 'window_meta.pkl'), 'wb') as f:
            pickle.dump(window_meta, f)
        print(f"Saved window_meta to {os.path.join(save_split_path, 'window_meta.pkl')}")
        # 스케일링 검증
        print("\n=== 스케일링 검증 ===")
        # sample_original = combined_data[:100]  # 처음 100개 샘플
        # sample_scaled = scaled_combined[:100]
        # sample_restored = scaler.inverse_transform(sample_scaled)
        sample_original = combined_data  # 처음 100개 샘플
        sample_scaled = scaled_combined
        sample_restored = scaler.inverse_transform(sample_scaled)
        feature_names = feature_cols
        verify_scaling(sample_original, sample_scaled, sample_restored, feature_names)

def verify_scaling(original_data, scaled_data, restored_data, feature_names):
    for i, feature in enumerate(feature_names):
        print(f"\n{feature} 검증:")
        print(f"원본 범위: {original_data[:, i].min():.3f} ~ {original_data[:, i].max():.3f}")
        print(f"스케일링 범위: {scaled_data[:, i].min():.3f} ~ {scaled_data[:, i].max():.3f}")
        print(f"복원 범위: {restored_data[:, i].min():.3f} ~ {restored_data[:, i].max():.3f}")
        print(f"MSE: {np.mean((original_data[:, i] - restored_data[:, i])**2):.6f}")
        
if __name__ == '__main__':
    feature_cols = [
        'Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time'
    ]
    
    load_all_valid_csv_tensors_by_cycle(
        folder_path="data_handling/yujin_files",
        feature_cols=feature_cols,
        batch_size=8,
        # save_split_path="./model/preprocessed_data_anomaly_eliminated",
        save_split_path="./model/preprocessed_data_test1",
        split_ratio=0.8,
        window_size=window_size,
        stride=window_size//4,
    )