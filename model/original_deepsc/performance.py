# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data, BatteryDataset, collate_battery_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags
import pdb
import joblib
import pandas as pd
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/test_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh2', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=5, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.__version__)  # PyTorch 버전 확인
print(torch.version.cuda)  # PyTorch에서 사용하는 CUDA 버전 확인

# using pre-trained model to compute the sentence similarity
class Similarity():
    def __init__(self, config_path, checkpoint_path, dict_path):
        self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
        self.model = keras.Model(inputs=self.model1.input,
                                 outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
        # build tokenizer
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def compute_similarity(self, real, predicted):
        token_ids1, segment_ids1 = [], []
        token_ids2, segment_ids2 = [], []
        score = []

        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            ids1, sids1 = self.tokenizer.encode(sent1)
            ids2, sids2 = self.tokenizer.encode(sent2)

            token_ids1.append(ids1)
            token_ids2.append(ids2)
            segment_ids1.append(sids1)
            segment_ids2.append(sids2)

        token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
        token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')

        segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
        segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')

        vector1 = self.model.predict([token_ids1, segment_ids1])
        vector2 = self.model.predict([token_ids2, segment_ids2])

        vector1 = np.sum(vector1, axis=1)
        vector2 = np.sum(vector2, axis=1)

        vector1 = normalize(vector1, axis=0, norm='max')
        vector2 = normalize(vector2, axis=0, norm='max')

        dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
        a = np.diag(np.matmul(vector1, vector1.T))  # a*a
        b = np.diag(np.matmul(vector2, vector2.T))

        a = np.sqrt(a)
        b = np.sqrt(b)

        output = dot / (a * b)
        score = output.tolist()

        return score


def performance(args, SNR, net):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)    

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []
            # pdb.set_trace()

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                # pdb.set_trace()
                Tx_word.append(word)
                Rx_word.append(target_word)

            # pdb.set_trace()
            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    score1 = np.mean(np.array(score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return score1#, score2

def evaluate_battery_performance(model, test_loader, scaler):
    model.eval()
    total_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_data = batch.to(device)
            output = model(input_data)
            
            # 실제 단위로 변환하여 MSE 계산
            input_original = scaler.inverse_transform(input_data.cpu().numpy())
            output_original = scaler.inverse_transform(output.cpu().numpy())
            
            mse = np.mean((input_original - output_original) ** 2)
            total_mse += mse * len(batch)
            total_samples += len(batch)
    
    return total_mse / total_samples

def train_step(net, input_data, target_data, noise_std, criterion, channel_type, mi_net=None):
    # input_data, target_data: (batch_size, window, feature)
    net.train()
    
    # 채널 노이즈 시뮬레이션
    if channel_type == 'AWGN':
        noise = torch.randn_like(input_data) * noise_std
        noisy_input = input_data + noise
    elif channel_type == 'Rayleigh':
        # Rayleigh 페이딩 구현
        pass
    
    # 모델 예측
    output = net(noisy_input)
    
    # 손실 계산
    loss = criterion(output, target_data)
    
    return loss.item()

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    # args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file
    args.vocab_file = 'C:/Users/ksshin/Desktop/ChanMinLee/DeepSC/DeepSC/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # 배터리 데이터용 파라미터
    input_dim = 6  # Voltage, Current, Temperature, Current_load, Voltage_load, Time
    window_size = 128  # 시계열 윈도우 크기

    # 모델 초기화
    deepsc = DeepSC(
        num_layers=args.num_layers,
        input_dim=input_dim,
        max_len=window_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        dropout=0.1
    ).to(device)

    # 데이터셋 변경
    train_dataset = BatteryDataset('train')
    test_dataset = BatteryDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_battery_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, collate_fn=collate_battery_data)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('model load!')

    bleu_score = performance(args, SNR, deepsc)
    print(bleu_score)

    #similarity.compute_similarity(sent1, real)

