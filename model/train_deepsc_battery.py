import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import joblib
from models.transceiver import DeepSC
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_deepsc_battery(
    train_pt='model/preprocessed_data/train_data.pt',
    test_pt='model/preprocessed_data/test_data.pt',
    scaler_path='model/preprocessed_data/scaler.pkl',
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
        dropout=0.1
    ).to(device)

    # 손실함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

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

        # (선택) 모델 저장
        torch.save(model.state_dict(), f"checkpoints/250621/deepsc_battery_epoch{epoch+1}.pth")

    print("학습 완료!")

if __name__ == "__main__":
    train_deepsc_battery() 