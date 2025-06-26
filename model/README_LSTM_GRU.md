# LSTM/GRU 기반 DeepSC 배터리 데이터 압축-복원

이 프로젝트는 기존 Transformer 기반 DeepSC 모델에 LSTM과 GRU 모델 옵션을 추가한 확장 버전입니다.

## 📁 파일 구조

```
model/
├── models/
│   ├── transceiver.py          # 기존 Transformer 기반 DeepSC 모델
│   ├── lstm_gru_models.py      # 새로운 LSTM/GRU 기반 모델들
│   └── mutual_info.py          # 상호 정보량 계산
├── test_deepsc_battery.py      # 모든 모델 테스트 및 비교 도구
├── train_lstm_gru_battery.py   # LSTM/GRU 모델 학습 도구
└── README_LSTM_GRU.md         # 이 파일
```

## 🚀 새로운 기능

### 1. 모델 옵션
- **Transformer**: 기존 DeepSC 모델 (Attention 기반)
- **LSTM**: Long Short-Term Memory 기반 모델
- **GRU**: Gated Recurrent Unit 기반 모델

### 2. 모델 아키텍처

#### LSTM DeepSC
- **인코더**: Bidirectional LSTM + 압축 레이어
- **디코더**: LSTM + 출력 레이어
- **특징**: 시계열의 장기 의존성 학습에 강점

#### GRU DeepSC
- **인코더**: Bidirectional GRU + 압축 레이어
- **디코더**: GRU + 출력 레이어
- **특징**: LSTM보다 간단한 구조, 빠른 학습

## 🛠️ 사용 방법

### 1. 모델 학습

```bash
cd model
python train_lstm_gru_battery.py
```

**옵션:**
- 1: LSTM 모델 학습
- 2: GRU 모델 학습
- 3: LSTM 모델 테스트
- 4: GRU 모델 테스트
- 5: LSTM 모델 학습 후 테스트
- 6: GRU 모델 학습 후 테스트

### 2. 모델 테스트 및 비교

```bash
cd model
python test_deepsc_battery.py
```

**모델 선택:**
- 1: Transformer 기반 DeepSC
- 2: LSTM 기반 DeepSC
- 3: GRU 기반 DeepSC
- 4: 모든 모델 비교

**기능 선택:**
- 5: 개별 window 압축-복원 테스트
- 6: 전체 배터리 시계열 복원
- 7: 원본 vs 복원 데이터 비교
- 8: 모든 기능 실행

## 📊 모델 비교

### 성능 지표
- **MSE (Mean Squared Error)**: 복원 품질 측정
- **압축률**: 원본 대비 압축된 데이터 크기 비율
- **압축 효율성**: (1 - 압축률) × 100%

### 예상 특징
- **Transformer**: 복잡한 패턴 학습에 우수, 높은 계산 비용
- **LSTM**: 장기 의존성 학습에 강점, 중간 계산 비용
- **GRU**: 빠른 학습, 간단한 구조, 효율적인 메모리 사용

## 🔧 하이퍼파라미터

### 공통 설정
- **입력 차원**: 6 (배터리 특성)
- **시퀀스 길이**: 128 (window 크기)
- **학습률**: 0.001
- **배치 크기**: 32
- **에포크**: 100

### LSTM/GRU 특정 설정
- **Hidden Dimension**: 128
- **Layer 수**: 2
- **Dropout**: 0.1
- **Bidirectional**: True (인코더만)

## 📈 결과 저장

### 체크포인트
```
checkpoints/250621/
├── deepsc_battery_epoch80.pth      # Transformer 모델
├── lstm_deepsc_battery_epoch80.pth # LSTM 모델
└── gru_deepsc_battery_epoch80.pth  # GRU 모델
```

### 복원 결과
```
reconstructed_transformer/  # Transformer 결과
reconstructed_lstm/         # LSTM 결과
reconstructed_gru/          # GRU 결과
```

### 시각화
```
model/
├── transformer_deepsc_battery_reconstruction.png
├── lstm_deepsc_battery_reconstruction.png
├── gru_deepsc_battery_reconstruction.png
├── lstm_training_curve.png
└── gru_training_curve.png
```

## 🎯 사용 시나리오

### 1. 모델 성능 비교
```python
# 모든 모델의 성능을 한 번에 비교
python test_deepsc_battery.py
# 옵션 4 선택
```

### 2. 특정 모델 학습
```python
# LSTM 모델 학습
python train_lstm_gru_battery.py
# 옵션 1 선택
```

### 3. 학습된 모델 테스트
```python
# GRU 모델 테스트
python test_deepsc_battery.py
# 옵션 3 (GRU) 선택 후 원하는 기능 선택
```

## ⚠️ 주의사항

1. **체크포인트 파일**: 학습된 모델이 없으면 랜덤 초기화된 모델로 테스트됩니다.
2. **GPU 메모리**: LSTM/GRU 모델은 Transformer보다 적은 메모리를 사용합니다.
3. **학습 시간**: GRU < LSTM < Transformer 순으로 학습 시간이 단축됩니다.
4. **데이터 전처리**: 기존 전처리된 데이터를 사용하므로 별도 전처리가 필요하지 않습니다.

## 🔍 모델 선택 가이드

### Transformer 선택 시기
- 복잡한 시계열 패턴이 있는 경우
- 충분한 계산 리소스가 있는 경우
- 최고 성능이 필요한 경우

### LSTM 선택 시기
- 장기 의존성이 중요한 경우
- 중간 수준의 계산 리소스가 있는 경우
- 안정적인 학습이 필요한 경우

### GRU 선택 시기
- 빠른 학습이 필요한 경우
- 제한된 계산 리소스가 있는 경우
- 간단하고 효율적인 모델이 필요한 경우

## 📝 향후 개선 사항

1. **하이퍼파라미터 튜닝**: 각 모델별 최적 파라미터 탐색
2. **앙상블 방법**: 여러 모델의 결과를 결합
3. **적응형 모델 선택**: 데이터 특성에 따른 자동 모델 선택
4. **실시간 압축**: 스트리밍 데이터에 대한 실시간 처리 