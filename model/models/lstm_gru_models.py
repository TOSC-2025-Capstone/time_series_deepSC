import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCompressor(nn.Module):
    """LSTM 기반 시계열 압축기"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(LSTMCompressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 압축을 위한 추가 레이어
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # bidirectional이므로 *2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)  # hidden_dim // 2에서 hidden_dim으로 변경
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 마지막 hidden state를 사용하여 압축
        # bidirectional이므로 forward와 backward의 마지막 hidden state를 결합
        last_forward = hidden[-2]  # forward direction의 마지막 hidden state
        last_backward = hidden[-1]  # backward direction의 마지막 hidden state
        combined = torch.cat([last_forward, last_backward], dim=1)
        
        compressed = self.compress(combined)
        return compressed

class LSTMDecompressor(nn.Module):
    """LSTM 기반 시계열 복원기"""
    def __init__(self, compressed_dim, hidden_dim, seq_len, output_dim, num_layers=2, dropout=0.1):
        super(LSTMDecompressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 압축된 벡터를 확장
        self.expand = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # LSTM을 통한 시계열 생성
        self.lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 시간별 가중치 레이어 추가
        self.time_weights = nn.Parameter(torch.randn(seq_len, hidden_dim * 2))
        
    def forward(self, compressed):
        # compressed shape: (batch, compressed_dim)
        batch_size = compressed.size(0)
        
        # 압축된 벡터를 확장
        expanded = self.expand(compressed)  # (batch, hidden_dim * 2)
        
        # 시간별로 다른 입력 시퀀스 생성 (단순 반복 대신)
        input_seq = expanded.unsqueeze(1) + self.time_weights.unsqueeze(0)  # (batch, seq_len, hidden_dim * 2)
        
        # LSTM을 통한 시계열 생성
        lstm_out, _ = self.lstm(input_seq)
        
        # 출력 레이어
        output = self.output_layer(lstm_out)  # (batch, seq_len, output_dim)
        
        return output

class LSTMDeepSC(nn.Module):
    """LSTM 기반 DeepSC 모델"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_layers=2, dropout=0.1):
        super(LSTMDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.compressed_dim = hidden_dim  # hidden_dim // 2에서 hidden_dim으로 변경
        
        self.encoder = LSTMCompressor(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecompressor(self.compressed_dim, hidden_dim, seq_len, input_dim, num_layers, dropout)
        
    def forward(self, x):
        # 압축
        compressed = self.encoder(x)
        # 복원
        reconstructed = self.decoder(compressed)
        return reconstructed
    
    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_dim
        return compressed_size / original_size

class GRUCompressor(nn.Module):
    """GRU 기반 시계열 압축기"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(GRUCompressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 압축을 위한 추가 레이어
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)  # hidden_dim // 2에서 hidden_dim으로 변경
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        gru_out, hidden = self.gru(x)
        
        # 마지막 hidden state를 사용하여 압축
        # bidirectional이므로 forward와 backward의 마지막 hidden state를 결합
        last_forward = hidden[-2]
        last_backward = hidden[-1]
        combined = torch.cat([last_forward, last_backward], dim=1)
        
        compressed = self.compress(combined)
        return compressed

class GRUDecompressor(nn.Module):
    """GRU 기반 시계열 복원기"""
    def __init__(self, compressed_dim, hidden_dim, seq_len, output_dim, num_layers=2, dropout=0.1):
        super(GRUDecompressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 압축된 벡터를 확장
        self.expand = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2)
        )
        
        # GRU를 통한 시계열 생성
        self.gru = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 출력 레이어
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 시간별 가중치 레이어 추가
        self.time_weights = nn.Parameter(torch.randn(seq_len, hidden_dim * 2))
        
    def forward(self, compressed):
        # compressed shape: (batch, compressed_dim)
        batch_size = compressed.size(0)
        
        # 압축된 벡터를 확장
        expanded = self.expand(compressed)  # (batch, hidden_dim * 2)
        
        # 시간별로 다른 입력 시퀀스 생성 (단순 반복 대신)
        input_seq = expanded.unsqueeze(1) + self.time_weights.unsqueeze(0)  # (batch, seq_len, hidden_dim * 2)
        
        # GRU를 통한 시계열 생성
        gru_out, _ = self.gru(input_seq)
        
        # 출력 레이어
        output = self.output_layer(gru_out)  # (batch, seq_len, output_dim)
        
        return output

class GRUDeepSC(nn.Module):
    """GRU 기반 DeepSC 모델"""
    def __init__(self, input_dim, seq_len, hidden_dim=128, num_layers=2, dropout=0.1):
        super(GRUDeepSC, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.compressed_dim = hidden_dim  # hidden_dim // 2에서 hidden_dim으로 변경
        
        self.encoder = GRUCompressor(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = GRUDecompressor(self.compressed_dim, hidden_dim, seq_len, input_dim, num_layers, dropout)
        
    def forward(self, x):
        # 압축
        compressed = self.encoder(x)
        # 복원
        reconstructed = self.decoder(compressed)
        return reconstructed
    
    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.compressed_dim
        return compressed_size / original_size

class Seq2SeqAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, num_layers=2, dropout=0.1):
        super(Seq2SeqAttention, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        # self.compressed_dim = hidden_dim//2 
        
        self.encoder = LSTMCompressor(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecompressor(hidden_dim, hidden_dim, seq_len, output_dim, num_layers, dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
    def forward(self, x):
        encoder_outputs = self.encoder(x)  # (batch, seq_len, hidden_dim)
        
        batch_size = x.size(0)
        seq_len = self.seq_len
        embed_dim = encoder_outputs.size(-1)  # 보통 hidden_dim*2

        # 제로 벡터로 디코더 입력 생성
        decoder_input = torch.zeros(batch_size, seq_len, embed_dim, device=x.device)

        # Attention 적용
        attn_output = self.attn(decoder_input, encoder_outputs, encoder_outputs)
        # 디코더에 attn_output 입력
        output = self.decoder(attn_output)
        return output 

    def get_compression_ratio(self):
        """압축률 계산"""
        original_size = self.input_dim * self.seq_len
        compressed_size = self.hidden_dim
        return compressed_size / original_size