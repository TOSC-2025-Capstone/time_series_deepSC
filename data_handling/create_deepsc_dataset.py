import pandas as pd
import numpy as np
import os
import torch
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from torch.utils.data import TensorDataset
import joblib

def load_selected_battery_data(folder_path, selected_batteries):
    """
    선택된 배터리 데이터만 로드하는 함수
    """
    all_data = []
    for battery in selected_batteries:
        file_name = f"B{battery:04d}.csv"
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Loaded {file_name}")
        else:
            print(f"File not found: {file_name}")
    
    return pd.concat(all_data, ignore_index=True)

def sliding_window_sequences(data, window_size=128, stride=64):
    """
    시계열 데이터를 슬라이딩 윈도우로 분할
    """
    sequences = []
    for start in range(0, len(data) - window_size + 1, stride):
        window = data[start:start + window_size]
        sequences.append(window)
    return np.stack(sequences)

def create_deepsc_dataset(input_folder, output_folder, selected_batteries, feature_cols, window_size=128, stride=64, split_ratio=0.8):
    """
    DeepSC 모델용 데이터셋 생성
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)
    
    # 데이터 로드
    print("Loading selected battery data...")
    df = load_selected_battery_data(input_folder, selected_batteries)
    
    # 데이터 전처리
    print("Preprocessing data...")
    scaler = MinMaxScaler()
    data = df[feature_cols].values.astype(np.float32)
    data = scaler.fit_transform(data)
    
    # 슬라이딩 윈도우 적용
    print("Creating sliding windows...")
    windows = sliding_window_sequences(data, window_size, stride)
    tensor_data = torch.tensor(windows, dtype=torch.float32)
    
    # 학습/테스트 데이터 분할
    N = tensor_data.shape[0]
    train_len = int(N * split_ratio)
    train_data = TensorDataset(tensor_data[:train_len])
    test_data = TensorDataset(tensor_data[train_len:])
    
    # 데이터 저장
    torch.save(train_data, os.path.join(output_folder, 'train_data.pt'))
    torch.save(test_data, os.path.join(output_folder, 'test_data.pt'))
    joblib.dump(scaler, os.path.join(output_folder, 'scaler.pkl'))
    
    print(f"Dataset created successfully:")
    print(f"   - Total samples: {N}")
    print(f"   - Train samples: {train_len}")
    print(f"   - Test samples: {N - train_len}")
    print(f"   - Features: {len(feature_cols)}")
    print(f"   - Window size: {window_size}")
    print(f"   - Stride: {stride}")

if __name__ == '__main__':
    # 설정
    INPUT_FOLDER = "merged"
    OUTPUT_FOLDER = "deepsc_dataset"
    SELECTED_BATTERIES = [5, 6, 7, 18, 33, 34, 36, 38, 39, 40]
    FEATURE_COLS = [
        'Voltage_measured', 'Current_measured', 'Temperature_measured',
        'Current_load', 'Voltage_load', 'Time'
    ]
    
    # 데이터셋 생성
    create_deepsc_dataset(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        selected_batteries=SELECTED_BATTERIES,
        feature_cols=FEATURE_COLS
    ) 