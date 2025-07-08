import torch
import torch.nn as nn
import joblib
import numpy as np
from tqdm import tqdm
from models.transceiver import DeepSC
from models.lstm_gru_models import LSTMDeepSC, GRUDeepSC
import matplotlib.pyplot as plt 
import pandas as pd
import os
import pickle
from collections import defaultdict

loss_type = 'MSE' # 3 loss 테스트 중 제일 좋았음
model_type = 'deepsc'
channel_type = 'no_channel'

def create_model(model_type, input_dim, window_size, device):
    """모델 타입에 따라 적절한 모델을 생성하는 함수"""
    if model_type == "deepsc":
        # 기존 Transformer 기반 DeepSC 모델
        model = DeepSC(
            num_layers=4,
            input_dim=input_dim,
            max_len=window_size,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1,
            compressed_len=64
        ).to(device)
        checkpoint_path = f'checkpoints/firstcase/{loss_type}/deepsc_battery_epoch79.pth'
        
    elif model_type == "lstm":
        # LSTM 기반 모델
        model = LSTMDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
        checkpoint_path = 'checkpoints/lstm_deepsc_battery/lstm_deepsc_battery_epoch20.pth'
        
    elif model_type == "gru":
        # GRU 기반 모델
        model = GRUDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
        checkpoint_path = 'checkpoints/gru_deepsc_battery/gru_deepsc_battery_epoch80.pth'
        
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    return model, checkpoint_path

def test_deepsc_battery(model_type="deepsc"):
    print(f"=== {model_type.upper()} 기반 배터리 데이터 압축-복원 검증 ===")
    
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
    
    # 3. 모델 초기화
    print(f"\n3. {model_type.upper()} 모델 초기화 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 하이퍼 파라미터 설정
    input_dim = train_tensor.shape[2]  # 6 features
    window_size = train_tensor.shape[1]  # 128
    
    try:
        # 모델 생성
        model, checkpoint_path = create_model(model_type, input_dim, window_size, device)
        
        # 학습된 파라미터 불러오기 (파일이 존재하는 경우에만)
        if os.path.exists(checkpoint_path):
            print(f"checkpoint_path: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"{model_type.upper()} 모델 초기화 및 파라미터 로드 성공: {checkpoint_path}")
        else:
            print(f"체크포인트 파일이 없습니다: {checkpoint_path}")
            print("랜덤 초기화된 모델로 테스트를 진행합니다.")
            
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
    save_dir = f'reconstructed_{"deepsc"}'
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
                # 모델을 통한 압축-복원 과정
                output = model(input_data)  # (1, window, feature)
                print(f"Sample {i}: Input shape: {input_data.shape}")
                print(f"Sample {i}: Output shape: {output.shape}")
                
                # 압축률 계산
                if model_type == "deepsc":
                    # Transformer 모델의 경우 기존 방식 사용
                    encoded = model.encoder(input_data, src_mask=None)
                    channel_encoded = model.channel_encoder(encoded)
                    compressed_size = channel_encoded.numel()
                else:
                    # LSTM/GRU 모델의 경우 모델의 압축률 사용
                    compression_ratio = model.get_compression_ratio()
                    compressed_size = int(input_data.numel() * compression_ratio)
                
                original_size = input_data.numel()
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
                    plt.savefig(f'model/{model_type}_deepsc_battery_reconstruction.png', dpi=300, bbox_inches='tight')
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
    
    print(f"\n=== {model_type.upper()} 검증 결과 ===")
    print(f"평균 MSE: {avg_mse:.6f}")
    print(f"평균 압축률: {avg_compression_ratio:.3f}")
    print(f"압축 효율성: {(1 - avg_compression_ratio) * 100:.1f}%")
    
    # 6. 성능 평가
    print(f"\n=== {model_type.upper()} 성능 평가 ===")
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

def reconstruct_battery_series(model_type="deepsc"):
    """
    전체 배터리 시계열을 복원하는 함수 (recon.py에서 가져온 기능)
    """
    print(f"=== {model_type.upper()} 기반 전체 배터리 시계열 복원 시작 ===")
    
    save_dir = f'reconstructed_{channel_type}_{model_type}_{loss_type}'
    # save_dir = f'reconstructed_{model_type}_{loss_type}'

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
    
    try:
        model, checkpoint_path = create_model(model_type, input_dim, window_size, device)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return

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
                plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed', alpha=0.7)
                plt.title(col)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} Comparison: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_path = os.path.join(save_dir, f'{base}_compare.png')
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"비교 그래프 저장: {fig_path}")
            
            # Residual(오차) 시계열 플롯 추가
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                residual = df_orig[col] - df_recon[col]
                plt.plot(residual, label='Residual', color='orange', alpha=0.8)
                plt.title(f'Residual: {col}')
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} Residuals: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            residual_fig_path = os.path.join(save_dir, f'{base}_residual.png')
            plt.savefig(residual_fig_path, dpi=200)
            plt.close()
            print(f"Residual 그래프 저장: {residual_fig_path}")

    # 6. 카운트 배열 확인
    for fname in battery_files:
        print(f"{fname} counts unique: {np.unique(counts[fname])}")

def compare_original_reconstructed(model_type="deepsc"):
    """
    원본 데이터와 복원 데이터를 비교하는 함수
    """
    print(f"=== {model_type.upper()} 원본 vs 복원 데이터 비교 ===")
    
    # 1. 원본 데이터 로드
    df_orig = pd.read_csv('data_handling/merged/B0005.csv')
    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # 2. 복원 데이터 로드 (예: 첫 window)
    save_dir = f'reconstructed_{model_type}'
    df_recon = pd.read_csv(f'{save_dir}/B0005_window0.csv')

    # 3. window의 시작 인덱스(예: 0)와 window_size(예: 128) 지정
    window_start = 0
    window_size = df_recon.shape[0]
    df_orig_window = df_orig[feature_cols].iloc[window_start:window_start+window_size].reset_index(drop=True)

    # 4. 비교 시각화
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i+1)
        plt.plot(df_orig_window[col], label='Original')
        plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed')
        plt.title(col)
        plt.legend()
        plt.grid(True)
    plt.suptitle(f'{model_type.upper()} vs Original Comparison')
    plt.tight_layout()
    plt.show()

def compare_all_models():
    """
    모든 모델의 성능을 비교하는 함수
    """
    print("=== 모든 모델 성능 비교 ===")
    
    models = ["deepsc", "lstm", "gru"]
    results = {}
    
    for model_type in models:
        print(f"\n{model_type.upper()} 모델 테스트 중...")
        try:
            # 간단한 테스트를 위해 1개 샘플만 사용
            test_data = torch.load('model/preprocessed_data/test_data.pt')
            test_tensor = test_data.tensors[0]
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            input_dim = test_tensor.shape[2]
            window_size = test_tensor.shape[1]
            
            model, checkpoint_path = create_model(model_type, input_dim, window_size, device)
            if os.path.exists(checkpoint_path):
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()
            
            # 테스트
            with torch.no_grad():
                input_data = test_tensor[0].unsqueeze(0).to(device)
                output = model(input_data)
                mse = nn.MSELoss()(output, input_data).item()
                
                if model_type == "deepsc":
                    encoded = model.encoder(input_data, src_mask=None)
                    channel_encoded = model.channel_encoder(encoded)
                    compressed_size = channel_encoded.numel()
                else:
                    compression_ratio = model.get_compression_ratio()
                    compressed_size = int(input_data.numel() * compression_ratio)
                
                original_size = input_data.numel()
                compression_ratio = compressed_size / original_size
                
                results[model_type] = {
                    'mse': mse,
                    'compression_ratio': compression_ratio,
                    'compression_efficiency': (1 - compression_ratio) * 100
                }
                
                print(f"{model_type.upper()}: MSE={mse:.6f}, 압축률={compression_ratio:.3f}, 효율성={(1-compression_ratio)*100:.1f}%")
                
        except Exception as e:
            print(f"{model_type.upper()} 모델 테스트 실패: {e}")
            results[model_type] = {'error': str(e)}
    
    # 결과 요약
    print(f"\n=== 모델 비교 결과 요약 ===")
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"{model_type.upper()}:")
            print(f"  MSE: {result['mse']:.6f}")
            print(f"  압축률: {result['compression_ratio']:.3f}")
            print(f"  압축 효율성: {result['compression_efficiency']:.1f}%")
        else:
            print(f"{model_type.upper()}: 오류 - {result['error']}")

if __name__ == "__main__":
    # 사용자가 선택할 수 있도록 메뉴 제공
    print("DeepSC 배터리 데이터 테스트 및 복원 도구")
    print("\n기능 선택:")
    print("1. 모든 모델 비교")
    print("2. 개별 window 압축-복원 테스트")
    print("3. 전체 배터리 시계열 복원")
    print("4. 원본 vs 복원 데이터 비교")
    print("5. 1개 모델의 2,3,4 모든 기능 실행")
    
    choice = input("원하는 옵션을 선택하세요 (1-5): ").strip()
    
    if choice == "1":
        compare_all_models()
        exit()
    elif choice == "2":
        model_choice = input("모델을 선택하세요 (deepsc/lstm/gru): ").strip().lower()
        if model_choice in ["deepsc", "lstm", "gru"]:
            test_deepsc_battery(model_choice)
        else:
            print("잘못된 모델 선택입니다. 기본값으로 deepsc를 사용합니다.")
            test_deepsc_battery("deepsc")
    elif choice == "3":
        model_choice = input("모델을 선택하세요 (deepsc/lstm/gru): ").strip().lower()
        if model_choice in ["deepsc", "lstm", "gru"]:
            reconstruct_battery_series(model_choice)
        else:
            print("잘못된 모델 선택입니다. 기본값으로 deepsc를 사용합니다.")
            reconstruct_battery_series("deepsc")
    elif choice == "4":
        model_choice = input("모델을 선택하세요 (deepsc/lstm/gru): ").strip().lower()
        if model_choice in ["deepsc", "lstm", "gru"]:
            compare_original_reconstructed(model_choice)
        else:
            print("잘못된 모델 선택입니다. 기본값으로 deepsc를 사용합니다.")
            compare_original_reconstructed("deepsc")
    elif choice == "5":
        model_choice = input("모델을 선택하세요 (deepsc/lstm/gru): ").strip().lower()
        if model_choice in ["deepsc", "lstm", "gru"]:
            test_deepsc_battery(model_choice)
            reconstruct_battery_series(model_choice)
            compare_original_reconstructed(model_choice)
        else:
            print("잘못된 모델 선택입니다. 기본값으로 deepsc를 사용합니다.")
            test_deepsc_battery("deepsc")
            reconstruct_battery_series("deepsc")
            compare_original_reconstructed("deepsc")
    else:
        print("잘못된 선택입니다. 기본값으로 deepsc 모델의 개별 window 테스트를 실행합니다.")
        test_deepsc_battery("deepsc") 