import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
from models.transceiver import DeepSC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pdb

def train_deepsc_battery(
    train_pt='model/preprocessed_data/train_data.pt',
    test_pt='model/preprocessed_data/test_data.pt',
    scaler_path='model/preprocessed_data/scaler.pkl',
    model_save_path='checkpoints/channel_case/Rician/deepsc_battery_epoch',
    # model_save_path='checkpoints/firstcase/MSE/deepsc_battery_epoch',
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

    pdb.set_trace()

    # val loss initial value 정의
    best_val_loss = 1

    # # early stopping
    # best_epoch_idx = 0

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_loss:.6f}")

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