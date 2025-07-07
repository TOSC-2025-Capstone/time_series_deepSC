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
from models.lstm_gru_models import LSTMDeepSC, GRUDeepSC, Seq2SeqAttention
import pdb

def train_improved_model(model_type, num_epochs=80, batch_size=32, learning_rate=1e-5):
    """
    개선된 LSTM 또는 GRU 모델을 학습하는 함수
    
    Args:
        model_type: "lstm" 또는 "gru"
        num_epochs: 학습 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
    """
    print(f"=== 개선된 {model_type.upper()} 모델 학습 시작 ===")
    
    # 1. 데이터 로드
    print("1. 데이터 로드 중...")
    train_data = torch.load('model/preprocessed_data/train_data.pt')
    val_data = torch.load('model/preprocessed_data/test_data.pt')
    
    train_tensor = train_data.tensors[0]
    val_tensor = val_data.tensors[0]
    
    print(f"Train data shape: {train_tensor.shape}")
    print(f"Validation data shape: {val_tensor.shape}")
    
    # 2. 데이터로더 생성
    train_dataset = TensorDataset(train_tensor, train_tensor)
    val_dataset = TensorDataset(val_tensor, val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. 모델 초기화
    print(f"2. 개선된 {model_type.upper()} 모델 초기화 중...")
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
    elif model_type == "seq2seq": # 테스트 중 
        model = Seq2SeqAttention(
            input_dim=input_dim,
            hidden_dim=128,
            seq_len=window_size,
            output_dim=input_dim,
            num_layers=2,
            dropout=0.1
        ).to(device)

    # 4. 손실 함수와 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)
    
    pdb.set_trace()

    # 5. 체크포인트 저장 디렉토리 생성
    checkpoint_dir = f'checkpoints/{model_type}_deepsc_battery'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 6. 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"3. 개선된 {model_type.upper()} 모델 학습 시작...")
    print(f"총 에포크: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"학습률: {learning_rate}")
    print(f"압축률: {model.get_compression_ratio():.3f}")
    
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
            
            # 그래디언트 클리핑 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
    plt.title(f'Improved {model_type.upper()} Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title(f'Improved {model_type.upper()} Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'model/improved_{model_type}_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"=== 개선된 {model_type.upper()} 모델 학습 완료 ===")
    print(f"최종 검증 손실: {best_val_loss:.6f}")
    print(f"최종 압축률: {compression_ratio:.3f}")
    print(f"최종 압축 효율성: {(1-compression_ratio)*100:.1f}%")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    print("개선된 LSTM/GRU 배터리 데이터 학습 도구")
    print("1. 개선된 LSTM 모델 학습")
    print("2. 개선된 GRU 모델 학습")
    print("3. Seq2Seq Attention 모델 학습")
    
    choice = input("원하는 기능을 선택하세요 (1-3): ").strip()
    
    if choice == "1":
        train_improved_model("lstm")
    elif choice == "2":
        train_improved_model("gru")
    elif choice == "3":
        train_improved_model("seq2seq")
    else:
        print("잘못된 선택입니다. 기본값으로 개선된 LSTM 모델 학습을 실행합니다.")
        train_improved_model("lstm") 