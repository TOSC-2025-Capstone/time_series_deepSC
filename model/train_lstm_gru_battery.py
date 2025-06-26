import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle
from models.lstm_gru_models import LSTMDeepSC, GRUDeepSC

def train_model(model_type, num_epochs=80, batch_size=32, learning_rate=1e-5):
    """
    LSTM 또는 GRU 모델을 학습하는 함수
    
    Args:
        model_type: "lstm" 또는 "gru"
        num_epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
    """
    print(f"=== {model_type.upper()} 모델 학습 시작 ===")
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    train_data = torch.load('model/preprocessed_data/train_data.pt')
    val_data = torch.load('model/preprocessed_data/test_data.pt')  # 검증용으로 test 데이터 사용
    
    train_tensor = train_data.tensors[0]  # (N, window, feature)
    val_tensor = val_data.tensors[0]
    
    print(f"Train data shape: {train_tensor.shape}")
    print(f"Validation data shape: {val_tensor.shape}")
    
    # 2. 데이터로더 생성
    train_dataset = TensorDataset(train_tensor, train_tensor)  # 자기회귀 학습
    val_dataset = TensorDataset(val_tensor, val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 모델 초기화
    print(f"2. {model_type.upper()} 모델 초기화 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    input_dim = train_tensor.shape[2]  # 6 features
    window_size = train_tensor.shape[1]  # 128
    
    if model_type == "lstm":
        model = LSTMDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    elif model_type == "gru":
        model = GRUDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    # 4. 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # 5. 체크포인트 저장 디렉토리 생성
    checkpoint_dir = f'checkpoints/250621'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 6. 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"3. {model_type.upper()} 모델 학습 시작...")
    print(f"총 에포크: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"학습률: {learning_rate}")
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        # 학습 루프
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_type}_deepsc_battery_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"새로운 최고 성능 모델 저장: {checkpoint_path}")
        
        # 진행 상황 출력
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.6f}')
        print(f'  Val Loss: {avg_val_loss:.6f}')
        print(f'  Best Val Loss: {best_val_loss:.6f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 압축률 정보 출력
        compression_ratio = model.get_compression_ratio()
        print(f'  압축률: {compression_ratio:.3f}')
        print(f'  압축 효율성: {(1-compression_ratio)*100:.1f}%')
        print('-' * 50)
        
        # 10 에포크마다 중간 체크포인트 저장
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_type}_deepsc_battery_epoch{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"중간 체크포인트 저장: {checkpoint_path}")
    
    # 7. 학습 곡선 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{model_type.upper()} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'{model_type.upper()} Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'model/{model_type}_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"=== {model_type.upper()} 모델 학습 완료 ===")
    print(f"최종 검증 손실: {best_val_loss:.6f}")
    print(f"최종 압축률: {compression_ratio:.3f}")
    print(f"최종 압축 효율성: {(1-compression_ratio)*100:.1f}%")
    
    return model, train_losses, val_losses

def test_trained_model(model_type):
    """
    학습된 모델을 테스트하는 함수
    """
    print(f"=== {model_type.upper()} 학습된 모델 테스트 ===")
    
    # 데이터 로드
    test_data = torch.load('model/preprocessed_data/test_data.pt')
    test_tensor = test_data.tensors[0]
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]
    window_size = test_tensor.shape[1]
    
    if model_type == "lstm":
        model = LSTMDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    elif model_type == "gru":
        model = GRUDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    
    # 최신 체크포인트 찾기
    checkpoint_dir = f'checkpoints/250621'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'{model_type}_deepsc_battery_epoch')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"체크포인트 로드: {checkpoint_path}")
    else:
        print("체크포인트 파일을 찾을 수 없습니다.")
        return
    
    model.eval()
    
    # 테스트
    with torch.no_grad():
        test_loss = 0.0
        num_samples = min(10, test_tensor.shape[0])
        
        for i in range(num_samples):
            input_data = test_tensor[i].unsqueeze(0).to(device)
            output = model(input_data)
            loss = nn.MSELoss()(output, input_data)
            test_loss += loss.item()
            
            if i == 0:  # 첫 번째 샘플 상세 분석
                print(f"첫 번째 샘플:")
                print(f"  입력 데이터 범위: {input_data.min():.6f} ~ {input_data.max():.6f}")
                print(f"  출력 데이터 범위: {output.min():.6f} ~ {output.max():.6f}")
                print(f"  MSE: {loss.item():.6f}")
        
        avg_test_loss = test_loss / num_samples
        compression_ratio = model.get_compression_ratio()
        
        print(f"\n=== {model_type.upper()} 테스트 결과 ===")
        print(f"평균 테스트 손실: {avg_test_loss:.6f}")
        print(f"압축률: {compression_ratio:.3f}")
        print(f"압축 효율성: {(1-compression_ratio)*100:.1f}%")

def reconstruct_and_compare_csv(model_type):
    """
    학습된 모델로 시계열을 복원하고 원본 CSV와 비교하는 함수
    """
    print(f"=== {model_type.upper()} 원본 vs 복원 CSV 비교 ===")
    
    # 1. 데이터 및 메타 정보 로드
    test_data = torch.load('model/preprocessed_data/test_data.pt')
    test_tensor = test_data.tensors[0]
    scaler = joblib.load('model/preprocessed_data/scaler.pkl')
    with open('model/preprocessed_data/window_meta.pkl', 'rb') as f:
        window_meta = pickle.load(f)
    train_data = torch.load('model/preprocessed_data/train_data.pt')
    train_len = len(train_data.tensors[0])

    # 2. 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = test_tensor.shape[2]
    window_size = test_tensor.shape[1]
    
    if model_type == "lstm":
        model = LSTMDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    elif model_type == "gru":
        model = GRUDeepSC(
            input_dim=input_dim,
            seq_len=window_size,
            hidden_dim=128,
            num_layers=2,
            dropout=0.1
        ).to(device)
    
    # 최신 체크포인트 찾기
    checkpoint_dir = f'checkpoints/250621'
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'{model_type}_deepsc_battery_epoch')]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('epoch')[1].split('.')[0]))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"체크포인트 로드: {checkpoint_path}")
    else:
        print("체크포인트 파일을 찾을 수 없습니다.")
        return
    
    model.eval()

    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # 3. 배터리별로 시계열 길이 추정 (원본 csv에서)
    battery_files = sorted(set([meta['file'] for meta in window_meta]))
    battery_lengths = {}
    for fname in battery_files:
        df = pd.read_csv(os.path.join('data_handling/merged', fname))
        battery_lengths[fname] = len(df)

    # 4. 배터리별로 빈 시계열 배열 준비 (복원값, 카운트)
    reconstructed = {fname: np.zeros((battery_lengths[fname], input_dim)) for fname in battery_files}
    counts = {fname: np.zeros(battery_lengths[fname]) for fname in battery_files}

    # 5. 각 window 복원 및 배터리별 시계열에 합치기
    print("시계열 복원 중...")
    with torch.no_grad():
        for i in tqdm(range(test_tensor.shape[0]), desc="Reconstructing"):
            input_data = test_tensor[i].unsqueeze(0).to(device)
            output = model(input_data)
            output_original = scaler.inverse_transform(output.squeeze(0).cpu().numpy())  # (window, feature)
            meta = window_meta[train_len + i]  # test set은 train 다음부터 시작
            fname = meta['file']
            start = meta['start']
            end = start + window_size
            reconstructed[fname][start:end] += output_original
            counts[fname][start:end] += 1

    # 6. 겹치는 부분 평균내기
    for fname in battery_files:
        mask = counts[fname] > 0
        reconstructed[fname][mask] /= counts[fname][mask][:, None]

    # 7. 배터리별로 csv 저장 및 비교 시각화
    save_dir = f'reconstructed_{model_type}'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== {model_type.upper()} 복원 결과 저장 및 비교 ===")
    for fname in battery_files:
        base = os.path.splitext(fname)[0]
        df_recon = pd.DataFrame(reconstructed[fname], columns=feature_cols)
        csv_path = os.path.join(save_dir, f'{base}_reconstructed.csv')
        df_recon.to_csv(csv_path, index=False)
        print(f"복원된 전체 시계열 저장: {csv_path}")

        # 비교 시각화 (test set에 포함된 배터리만)
        if np.any(counts[fname] > 0):
            df_orig = pd.read_csv(os.path.join('data_handling/merged', fname))
            
            # 시각화
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(feature_cols):
                plt.subplot(2, 3, i+1)
                plt.plot(df_orig[col], label='Original', alpha=0.7, linewidth=1)
                plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed', alpha=0.7, linewidth=1)
                plt.title(col)
                plt.legend()
                plt.grid(True)
            plt.suptitle(f'{model_type.upper()} vs Original: {fname}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig_path = os.path.join(save_dir, f'{base}_compare.png')
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"비교 그래프 저장: {fig_path}")
            
            # 통계 정보 출력
            print(f"\n=== {fname} 통계 정보 ===")
            for col in feature_cols:
                orig_mean = df_orig[col].mean()
                recon_mean = df_recon[col].mean()
                orig_std = df_orig[col].std()
                recon_std = df_recon[col].std()
                mse = np.mean((df_orig[col] - df_recon[col]) ** 2)
                mae = np.mean(np.abs(df_orig[col] - df_recon[col]))
                
                print(f"{col}:")
                print(f"  원본 - 평균: {orig_mean:.4f}, 표준편차: {orig_std:.4f}")
                print(f"  복원 - 평균: {recon_mean:.4f}, 표준편차: {recon_std:.4f}")
                print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}")
                print(f"  평균 차이: {abs(orig_mean - recon_mean):.6f}")
                print(f"  표준편차 차이: {abs(orig_std - recon_std):.6f}")

    # 8. 카운트 배열 확인
    print(f"\n=== 복원 범위 정보 ===")
    for fname in battery_files:
        unique_counts = np.unique(counts[fname])
        print(f"{fname}:")
        print(f"  복원된 구간 수: {len(unique_counts)}")
        print(f"  최대 중복 횟수: {unique_counts.max()}")
        print(f"  복원된 데이터 포인트: {np.sum(counts[fname] > 0)} / {len(counts[fname])}")

def compare_single_window(model_type):
    """
    단일 window의 원본 vs 복원 비교
    """
    print(f"=== {model_type.upper()} 단일 Window 비교 ===")
    
    # 1. 원본 데이터 로드
    df_orig = pd.read_csv('data_handling/merged/B0005.csv')
    feature_cols = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load', 'Time']

    # 2. 복원 데이터 로드 (첫 번째 window)
    save_dir = f'reconstructed_{model_type}'
    if os.path.exists(f'{save_dir}/B0005_window0.csv'):
        df_recon = pd.read_csv(f'{save_dir}/B0005_window0.csv')
    else:
        print(f"복원된 window 파일을 찾을 수 없습니다: {save_dir}/B0005_window0.csv")
        return

    # 3. window의 시작 인덱스(0)와 window_size(128) 지정
    window_start = 0
    window_size = df_recon.shape[0]
    df_orig_window = df_orig[feature_cols].iloc[window_start:window_start+window_size].reset_index(drop=True)

    # 4. 상세 비교 시각화
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(feature_cols):
        plt.subplot(2, 3, i+1)
        plt.plot(df_orig_window[col], label='Original', linewidth=2, alpha=0.8)
        plt.plot(df_recon[col], label=f'{model_type.upper()} Reconstructed', linewidth=2, alpha=0.8)
        plt.title(f'{col}\nWindow {window_start}-{window_start+window_size}')
        plt.legend()
        plt.grid(True)
        
        # MSE 계산 및 표시
        mse = np.mean((df_orig_window[col] - df_recon[col]) ** 2)
        plt.text(0.02, 0.98, f'MSE: {mse:.6f}', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{model_type.upper()} vs Original - Single Window Comparison')
    plt.tight_layout()
    plt.savefig(f'model/{model_type}_single_window_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. 수치적 비교 결과 출력
    print(f"\n=== {model_type.upper()} 단일 Window 수치 비교 ===")
    for col in feature_cols:
        orig_data = df_orig_window[col]
        recon_data = df_recon[col]
        
        mse = np.mean((orig_data - recon_data) ** 2)
        mae = np.mean(np.abs(orig_data - recon_data))
        corr = np.corrcoef(orig_data, recon_data)[0, 1]
        
        print(f"{col}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  상관계수: {corr:.6f}")
        print(f"  원본 범위: {orig_data.min():.4f} ~ {orig_data.max():.4f}")
        print(f"  복원 범위: {recon_data.min():.4f} ~ {recon_data.max():.4f}")
        print()

if __name__ == "__main__":
    print("LSTM/GRU 배터리 데이터 학습 및 비교 도구")
    print("1. LSTM 모델 학습")
    print("2. GRU 모델 학습")
    print("3. LSTM 모델 테스트")
    print("4. GRU 모델 테스트")
    print("5. LSTM 모델 학습 후 테스트")
    print("6. GRU 모델 학습 후 테스트")
    print("7. LSTM 모델 CSV 비교 (전체 시계열)")
    print("8. GRU 모델 CSV 비교 (전체 시계열)")
    print("9. LSTM 모델 단일 Window 비교")
    print("10. GRU 모델 단일 Window 비교")
    
    choice = input("원하는 기능을 선택하세요 (1-10): ").strip()
    
    if choice == "1":
        train_model("lstm")
    elif choice == "2":
        train_model("gru")
    elif choice == "3":
        test_trained_model("lstm")
    elif choice == "4":
        test_trained_model("gru")
    elif choice == "5":
        train_model("lstm")
        test_trained_model("lstm")
    elif choice == "6":
        train_model("gru")
        test_trained_model("gru")
    elif choice == "7":
        reconstruct_and_compare_csv("lstm")
    elif choice == "8":
        reconstruct_and_compare_csv("gru")
    elif choice == "9":
        compare_single_window("lstm")
    elif choice == "10":
        compare_single_window("gru")
    else:
        print("잘못된 선택입니다. 기본값으로 LSTM 모델 학습을 실행합니다.")
        train_model("lstm") 