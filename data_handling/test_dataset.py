import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

def load_scaler():
    scaler = joblib.load('deepsc_dataset/scaler.pkl')
    return scaler

def load_dataset():
    train_data = torch.load('deepsc_dataset/train_data.pt')
    test_data = torch.load('deepsc_dataset/test_data.pt')
    return train_data, test_data

def inverse_transform_data(data, scaler):
    # 데이터를 numpy 배열로 변환
    data_np = data.numpy()
    # 역정규화 수행
    original_data = scaler.inverse_transform(data_np)
    return original_data

def plot_sample(data, title, scaler):
    sample = data[0]  # (시계열 길이, 특성 수) 또는 (특성 수, 시계열 길이)
    # shape이 (특성 수, 시계열 길이)라면 transpose 필요
    if sample.shape[0] < sample.shape[1]:
        sample = sample.T
    original_sample = scaler.inverse_transform(sample)
    plt.figure(figsize=(12, 6))
    for i in range(original_sample.shape[1]):
        plt.plot(original_sample[:, i], label=f'Feature {i}')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 스케일러 로드
    scaler = load_scaler()
    
    # 데이터셋 로드
    train_data, test_data = load_dataset()
    
    # 데이터셋 정보 출력
    print("Train data shape:", train_data.tensors[0].shape)
    print("Test data shape:", test_data.tensors[0].shape)
    
    # 데이터 샘플 시각화
    plot_sample(train_data.tensors[0], "Train Data Sample (First Sequence)", scaler)
    plot_sample(test_data.tensors[0], "Test Data Sample (First Sequence)", scaler)
    
    # 데이터 통계 정보 출력
    print("\nTrain data statistics:")
    print("Mean:", train_data.tensors[0].mean().item())
    print("Std:", train_data.tensors[0].std().item())
    print("Min:", train_data.tensors[0].min().item())
    print("Max:", train_data.tensors[0].max().item())
    
    print("\nTest data statistics:")
    print("Mean:", test_data.tensors[0].mean().item())
    print("Std:", test_data.tensors[0].std().item())
    print("Min:", test_data.tensors[0].min().item())
    print("Max:", test_data.tensors[0].max().item())

if __name__ == "__main__":
    main() 