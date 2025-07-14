import numpy as np
import torch
from ts2vec import TS2Vec

# 예시: 데이터 로딩 (여기에 실제 데이터 로딩 코드를 넣으세요)
# 예시: (batch_size, seq_len, feature)
# 아래는 임시 더미 데이터입니다. 실제 데이터로 교체하세요.
# 예: np.load, pd.read_csv 등으로 데이터 불러오기
batch_size = 32
seq_len = 128
feature = 6
train_data = np.random.randn(batch_size, seq_len, feature).astype(np.float32)
test_data = np.random.randn(batch_size, seq_len, feature).astype(np.float32)

# TS2Vec 모델 정의
model = TS2Vec(
    input_dims=feature,   # 피쳐 수
    device='cuda' if torch.cuda.is_available() else 'cpu',
    output_dims=16        # 압축 차원(원하는 대로 조정)
)

# 학습
print('Fitting TS2Vec...')
loss_log = model.fit(
    train_data,
    verbose=True
)

# 인스턴스별 representation 추출 (압축)
print('Encoding...')
compressed = model.encode(test_data, encoding_window='full_series')  # (batch_size, output_dims)

# 시점별 representation 추출 (압축)
compressed_seq = model.encode(test_data)  # (batch_size, seq_len, output_dims)

print('Compressed shape (instance-level):', compressed.shape)
print('Compressed shape (timestamp-level):', compressed_seq.shape)

# 복원(디코딩) 기능은 TS2Vec 기본에는 없음.
# 하지만, compressed representation을 decoder(LSTM, MLP 등)에 넣어 복원 네트워크를 추가로 만들 수 있음.
# 아래는 예시로, 간단한 MLP decoder를 붙여 복원하는 코드입니다.

import torch.nn as nn

class SimpleDecoder(nn.Module):
    def __init__(self, compressed_dim, seq_len, feature):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(compressed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * feature)
        )
        self.seq_len = seq_len
        self.feature = feature

    def forward(self, x):
        out = self.fc(x)
        return out.view(-1, self.seq_len, self.feature)

# 인스턴스별 압축 벡터로 복원
decoder = SimpleDecoder(compressed_dim=compressed.shape[1], seq_len=seq_len, feature=feature)
decoder = decoder.to(compressed.device if hasattr(compressed, 'device') else 'cpu')

compressed_tensor = torch.tensor(compressed, dtype=torch.float32)
reconstructed = decoder(compressed_tensor)

print('Reconstructed shape:', reconstructed.shape)

# 복원 성능 평가 (예: MSE)
mse = ((reconstructed.detach().cpu().numpy() - test_data) ** 2).mean()
print('Reconstruction MSE:', mse)