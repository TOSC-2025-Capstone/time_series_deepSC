import torch
import torch.nn as nn
import joblib
import numpy as np
from tqdm import tqdm
from models.transceiver import DeepSC
import matplotlib.pyplot as plt 
import pandas as pd
import os
import pickle

def test_deepsc_battery():
    print("=== DeepSC 배터리 데이터 압축-복원 검증 ===")
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    train_data = torch.load('model/preprocessed_data/train_data.pt')
    test_data = torch.load('model/preprocessed_data/test_data.pt')
    scaler = joblib.load('model/preprocessed_data/scaler.pkl')
    
    train_tensor = train_data.tensors[0]  # (N, window, feature)
    test_tensor = test_data.tensors[0]
    
    print(f"Train data shape: {train_tensor.shape}")
    print(f"Test data shape: {test_tensor.shape}")
    print(f"Feature dimension: {train_tensor.shape[2]}")
    print(f"Window size: {train_tensor.shape[1]}")
    
    # 2. 데이터 범위 확인
    print(f"\n2. 데이터 범위 확인:")
    print(f"Train data range: {train_tensor.min():.6f} ~ {train_tensor.max():.6f}")
    print(f"Test data range: {test_tensor.min():.6f} ~ {test_tensor.max():.6f}")
    
    # 3. DeepSC 모델 초기화
    print(f"\n3. DeepSC 모델 초기화 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 하이퍼 파라미터 설정
    input_dim = train_tensor.shape[2]  # 6 features
    window_size = train_tensor.shape[1]  # 128
    d_model = 128
    num_heads = 8
    num_layers = 4
    dff = 512
    dropout = 0.1
    
    try:
        # 수정된 DeepSC 모델 사용 (시계열 데이터용)
        model = DeepSC(
            num_layers=num_layers,
            input_dim=input_dim,  # 6 features
            max_len=window_size,  # 128
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout=dropout
        ).to(device)
        # 학습된 파라미터 불러오기
        checkpoint_path = 'checkpoints/250621/deepsc_battery_epoch80.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"DeepSC 모델 초기화 및 파라미터 로드 성공: {checkpoint_path}")
    except Exception as e:
        print(f"모델 초기화 실패: {e}")
        return
    
    # 4. 간단한 압축-복원 테스트
    print(f"\n4. 압축-복원 테스트 중...")
    model.eval()
    
    # 테스트할 샘플 수
    num_test_samples = min(5, test_tensor.shape[0])
    
    total_mse = 0
    total_compression_ratio = 0
    
    # 복원된 시계열을 csv로 저장할 폴더
    save_dir = 'reconstructed'
    os.makedirs(save_dir, exist_ok=True)
    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # window_meta 불러오기
    with open('model/preprocessed_data/window_meta.pkl', 'rb') as f:
        window_meta = pickle.load(f)

    with torch.no_grad():
        for i in tqdm(range(num_test_samples), desc="Testing samples"):
            # 입력 데이터 준비
            input_data = test_tensor[i].unsqueeze(0).to(device)  # (1, window, feature)
            
            try:
                # 모델을 통한 압축-복원 과정 (전체 모델 한 번에 통과)
                output = model(input_data)  # (1, window, feature)
                print(f"Sample {i}: Input shape: {input_data.shape}")
                print(f"Sample {i}: Output shape: {output.shape}")
                
                # 압축률 계산 (채널 인코더 출력 기준)
                encoded = model.encoder(input_data, src_mask=None)
                channel_encoded = model.channel_encoder(encoded)
                
                original_size = input_data.numel()
                compressed_size = channel_encoded.numel()
                compression_ratio = compressed_size / original_size
                total_compression_ratio += compression_ratio
                
                # MSE 계산
                mse = nn.MSELoss()(output, input_data)
                total_mse += mse.item()
                
                print(f"Sample {i}: MSE = {mse.item():.6f}, Compression ratio = {compression_ratio:.3f}")
                
                # 첫 번째 샘플 상세 분석
                if i == 0:
                    print(f"\n=== 첫 번째 샘플 상세 분석 ===")
                    print(f"입력 데이터 범위: {input_data.min():.6f} ~ {input_data.max():.6f}")
                    print(f"출력 데이터 범위: {output.min():.6f} ~ {output.max():.6f}")
                    
                    # 실제 단위로 변환하여 비교
                    input_original = scaler.inverse_transform(input_data.squeeze(0).cpu().numpy())
                    output_original = scaler.inverse_transform(output.squeeze(0).cpu().numpy())
                    
                    mse_original = np.mean((input_original - output_original) ** 2)
                    print(f"실제 단위 MSE: {mse_original:.6f}")
                    
                    # 시각화
                    plt.figure(figsize=(15, 10))
                    
                    # 첫 번째 특성 (전압) 비교
                    plt.subplot(2, 3, 1)
                    plt.plot(input_original[:, 0], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 0], label='Reconstructed', alpha=0.7)
                    plt.title('Voltage_measured')
                    plt.legend()
                    plt.grid(True)
                    
                    # 두 번째 특성 (전류) 비교
                    plt.subplot(2, 3, 2)
                    plt.plot(input_original[:, 1], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 1], label='Reconstructed', alpha=0.7)
                    plt.title('Current_measured')
                    plt.legend()
                    plt.grid(True)
                    
                    # 세 번째 특성 (온도) 비교
                    plt.subplot(2, 3, 3)
                    plt.plot(input_original[:, 2], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 2], label='Reconstructed', alpha=0.7)
                    plt.title('Temperature_measured')
                    plt.legend()
                    plt.grid(True)
                    
                    # 네 번째 특성 (부하 전류) 비교
                    plt.subplot(2, 3, 4)
                    plt.plot(input_original[:, 3], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 3], label='Reconstructed', alpha=0.7)
                    plt.title('Current_load')
                    plt.legend()
                    plt.grid(True)
                    
                    # 다섯 번째 특성 (부하 전압) 비교
                    plt.subplot(2, 3, 5)
                    plt.plot(input_original[:, 4], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 4], label='Reconstructed', alpha=0.7)
                    plt.title('Voltage_load')
                    plt.legend()
                    plt.grid(True)
                    
                    # 여섯 번째 특성 (시간) 비교
                    plt.subplot(2, 3, 6)
                    plt.plot(input_original[:, 5], label='Original', alpha=0.7)
                    plt.plot(output_original[:, 5], label='Reconstructed', alpha=0.7)
                    plt.title('Time')
                    plt.legend()
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.savefig('model/deepsc_battery_reconstruction.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                # 복원 시계열을 실제 단위로 변환
                output_original = scaler.inverse_transform(output.squeeze(0).cpu().numpy())
                # csv로 저장
                meta = window_meta[i]
                base_name = os.path.splitext(meta['file'])[0]  # B0005 등
                start_idx = meta['start']
                csv_path = os.path.join(save_dir, f'{base_name}_window{start_idx}.csv')
                pd.DataFrame(output_original, columns=feature_cols).to_csv(csv_path, index=False)
                print(f"복원된 시계열 저장: {csv_path}")
                
            except Exception as e:
                print(f"샘플 {i} 처리 중 오류: {e}")
                continue
    
    # 5. 결과 출력
    avg_mse = total_mse / num_test_samples
    avg_compression_ratio = total_compression_ratio / num_test_samples
    
    print(f"\n=== 검증 결과 ===")
    print(f"평균 MSE: {avg_mse:.6f}")
    print(f"평균 압축률: {avg_compression_ratio:.3f}")
    print(f"압축 효율성: {(1 - avg_compression_ratio) * 100:.1f}%")
    
    # 6. 성능 평가
    print(f"\n=== 성능 평가 ===")
    if avg_mse < 0.01:
        print("우수한 복원 성능: MSE가 매우 낮습니다.")
    elif avg_mse < 0.1:
        print("양호한 복원 성능: MSE가 낮습니다.")
    elif avg_mse < 1.0:
        print("보통 복원 성능: MSE가 중간 수준입니다.")
    else:
        print("낮은 복원 성능: MSE가 높습니다. 모델 개선이 필요합니다.")
    
    if avg_compression_ratio < 0.1:
        print("우수한 압축 효율성: 90% 이상 압축되었습니다.")
    elif avg_compression_ratio < 0.2:
        print("양호한 압축 효율성: 80% 이상 압축되었습니다.")
    elif avg_compression_ratio < 0.5:
        print("보통 압축 효율성: 50% 이상 압축되었습니다.")
    else:
        print("낮은 압축 효율성: 압축률이 낮습니다.")

if __name__ == "__main__":
    test_deepsc_battery()

# 1. 원본 데이터 로드
df_orig = pd.read_csv('data_handling/merged/B0005.csv')
feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

# 2. 복원 데이터 로드 (예: 첫 window)
df_recon = pd.read_csv('reconstructed/B0005_window0.csv')

# 3. window의 시작 인덱스(예: 0)와 window_size(예: 128) 지정
window_start = 0
window_size = df_recon.shape[0]
df_orig_window = df_orig[feature_cols].iloc[window_start:window_start+window_size].reset_index(drop=True)

# 4. 비교 시각화
plt.figure(figsize=(15, 10))
for i, col in enumerate(feature_cols):
    plt.subplot(2, 3, i+1)
    plt.plot(df_orig_window[col], label='Original')
    plt.plot(df_recon[col], label='Reconstructed')
    plt.title(col)
    plt.legend()
    plt.grid(True)
plt.tight_layout()
plt.show() 