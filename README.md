### time_series_deepSC

## 프로젝트 실행 방법
1. zip 다운로드 후 colab에서 프로젝트로 등록
2. requirements.txt대로 라이브러리 설치하기 위해서 !pip install -r requirements.txt 명령어 실행
3. 이후 다음 순서로 코드 실행
    create_deepsc_dataset.py : 데이터 생성
    - DeepSC 테스트일 경우
    train_deepsc_battery.py
        모델 저장 경로(line 80 근처)는 본인이 원하는 곳으로 설정후 테스트에서 경로 불러오는 코드(line 14~)와 맞추기 
    test_deepsc_battery.py -> 옵션 2,3,4,5 -> deepsc
    - LSTM, GRU 테스트일 경우
    train_improved_lstm_gru.py
    test_deepsc_battery.py -> 옵션 2,3,4,5 -> lstm, gru