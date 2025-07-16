import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
from models.transceiver import DeepSC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt 

def train_deepsc_battery(
    train_pt='model/preprocessed_data_0715/train_data.pt',
    test_pt='model/preprocessed_data_0715/test_data.pt',
    scaler_path='model/preprocessed_data_0715/scaler.pkl',
    # model_save_path='checkpoints/channel_case/Rician/deepsc_battery_epoch',
    model_save_path='checkpoints/case3/MSE/deepsc/deepsc_battery_epoch',
    num_epochs=80,
    batch_size=32,
    lr=1e-5,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로드
    train_data = torch.load(train_pt)
    test_data = torch.load(test_pt)
    train_tensor = train_data.tensors[0]
    test_tensor = test_data.tensors[0]
    scaler = joblib.load(scaler_path)

    # DataLoader
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    # 모델 초기화
    input_dim = train_tensor.shape[2]
    window_size = train_tensor.shape[1]
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

    # 손실함수 및 옵티마이저
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # val loss initial value 정의
    best_val_loss = 1

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [DeepSC-Train]')
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.6f}")

        # === 정규화된 입력과 output 비교 plot (첫 배치만) ===
        if epoch == 10:
            # batch: [batch_size, window, feature]
            input_norm = batch[:6, :, :6].detach().cpu().numpy()   # [6, window, 6]
            output_norm = output[:6, :, :6].detach().cpu().numpy() # [6, window, 6]
            for sample_idx in range(6):
                plt.figure(figsize=(15, 8))
                for i in range(input_norm.shape[2]):
                    plt.subplot(2, 3, i+1)
                    plt.plot(input_norm[sample_idx, :, i], label='Input (norm)', color='blue', alpha=0.7)
                    plt.plot(output_norm[sample_idx, :, i], label='Output (norm)', color='orange', alpha=0.7)
                    plt.title(f'Feature {i+1}')
                    plt.legend()
                    plt.grid(True)
                plt.suptitle(f'정규화 입력 vs Output (Epoch {epoch+1}, Sample {sample_idx+1})')
                plt.tight_layout()
                plt.savefig(f'model/train_input_vs_output_norm_epoch{epoch+1}_sample{sample_idx+1}.png', dpi=200)
                plt.show()

        # 검증
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item() * batch.size(0)
        avg_val_loss = val_loss / len(test_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {avg_val_loss:.6f}")

        # 스케줄러 step (val loss 기준)
        scheduler.step(avg_val_loss)

        # val loss 개선 시 모델 저장
        if(avg_val_loss < best_val_loss):
            torch.save(model.state_dict(), model_save_path+f"{epoch+1}.pth")
            best_val_loss = avg_val_loss
            best_epoch_idx = epoch
            print(f"[Best Val Epoch {epoch+1}/{num_epochs}] Best Val Loss: {best_val_loss}")  
        
        # # early stopping - patience = 10
        # if(epoch > best_epoch_idx+10):
        #     print("조기 중단에 의한 학습 완료!")
        #     return
    print("학습 완료!")

if __name__ == "__main__":
    train_deepsc_battery() 