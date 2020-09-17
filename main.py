"""
Copyright 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#-*- coding: utf-8 -*-
import numpy as np
import os
import sys
import time
import math
import wavio
import argparse
import queue
import shutil
import random
import math
import time
import torch 
import logging
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import Levenshtein as Lev 

import label_loader
from loader import *
from models import EncoderRNN, DecoderRNN, Seq2seq

import config

char2index = dict() # 딕셔너리 선언
index2char = dict() # 딕셔너리 선언
SOS_token = 0
EOS_token = 0
PAD_token = 0

#DATASET_PATH = os.path.join(DATASET_PATH, 'train') # 경로를 병합하여 새 경로 생성 DATASET_PATH + 'train'

def label_to_string(labels):
    if len(labels.shape) == 1: # 라벨 shape 길이가 1이면
        sent = str() # 숫자를 문자열로 변화시키는 함수
        for i in labels:
            if i.item() == EOS_token:
                break
            sent += index2char[i.item()]
        return sent

    elif len(labels.shape) == 2: # 라벨 shape 길이가 1이면
        sents = list() # 비어 있는 리스트 만듬
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents

def char_distance(ref, hyp):
    ref = ref.replace(' ', '') 
    hyp = hyp.replace(' ', '') 

    dist = Lev.distance(hyp, ref) # distance 글자 몇개 틀렸는지 확인
    length = len(ref.replace(' ', ''))

    return dist, length 

def get_distance(ref_labels, hyp_labels, display=False): # 이게 cer 구하는것 리더보드 올릴때
    total_dist = 0
    total_length = 0
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length 
        if display:
            cer = total_dist / total_length
            logger.debug('%d (%0.4f)\n(%s)\n(%s)' % (i, cer, ref, hyp))
    return total_dist, total_length

def train(model, total_batch_size, queue, criterion, optimizer, device, train_begin, train_loader_count, print_batch=5, teacher_forcing_ratio=1):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    batch = 0

    model.train() # 모델(seq2seq) train 모델을 학습모드로

    logger.info('train() start')

    begin = epoch_begin = time.time() # 지금 시간
    while True:
        if queue.empty(): # 큐가 비어 있으면 True를, 그렇지 않으면 False를 반환합니다
            logger.debug('queue is empty')

        feats, scripts, feat_lengths, script_lengths = queue.get() # 큐에서 항목을 제거하고 반환합니다.
        # feat -> stft한 값 / script -> 정답 값

        if feats.shape[0] == 0: 
            # empty feats means closing one loader
            train_loader_count -= 1

            logger.debug('left train_loader: %d' % (train_loader_count))

            if train_loader_count == 0: # train_loader_count 가 0이 되면 while문 빠져나옴
                break
            else:
                continue
        
        optimizer.zero_grad() # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.

        feats = feats.to(device) 
        scripts = scripts.to(device)

        src_len = scripts.size(1)

        target = scripts[:, 1:] # 맨 앞은 818 (<s>) 로 똑같음

        model.module.flatten_parameters()

        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio) # 이게 포워드 함수 실행하는것 같은데?
       
        logit = torch.stack(logit, dim=1).to(device) # concaternate 사용
       
        y_hat = logit.max(-1)[1] 
        # max output -> 값 , indices / y_hat 은 indices

        loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1)) # 오차 계산 <-- 이게 말단
        # logit view는 (1, 2, 3) ->를 (1 x 2, 3)으로 만들어 줌
        total_loss += loss.item() # 누적 오차 계산 loss의 스칼라 값  loss는 (1,) 형태의 Tensor이며, loss.item()은 loss의 스칼라 값입니다.
        total_num += sum(feat_lengths)

        display = random.randrange(0, 100) == 0

        dist, length = get_distance(target, y_hat, display=display) # cer 구하기 위한 단계
        # y_hat은 max index
        total_dist += dist
        total_length += length
        # 요 위에 있는 애들 cer 구하는거임

        total_sent_num += target.size(0)

        loss.backward() # backward 함수 계산
        optimizer.step() # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.

        if batch % print_batch == 0: # 진행상황 출력
            current = time.time()
            elapsed = current - begin
            epoch_elapsed = (current - epoch_begin) / 60.0
            train_elapsed = (current - train_begin) / 3600.0

            logger.info('batch: {:4d}/{:4d}, loss: {:.4f}, cer: {:.2f}, elapsed: {:.2f}s {:.2f}m {:.2f}h'
                .format(batch,
                        #len(dataloader),
                        total_batch_size,
                        total_loss / total_num,
                        total_dist / total_length,
                        elapsed, epoch_elapsed, train_elapsed))
            begin = time.time()

            
        batch += 1
        train.cumulative_batch_count += 1

    logger.info('train() completed')
    return total_loss / total_num, total_dist / total_length

train.cumulative_batch_count = 0

def evaluate(model, dataloader, queue, criterion, device):
    logger.info('evaluate() start')
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.eval()

    with torch.no_grad():
        while True:
            feats, scripts, feat_lengths, script_lengths = queue.get()
            if feats.shape[0] == 0:
                break

            feats = feats.to(device)
            scripts = scripts.to(device)

            src_len = scripts.size(1)
            target = scripts[:, 1:]

            model.module.flatten_parameters()
            logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=0.0)

            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths)

            display = random.randrange(0, 100) == 0
            dist, length = get_distance(target, y_hat, display=display)
            total_dist += dist
            total_length += length
            total_sent_num += target.size(0)

    logger.info('evaluate() completed')
    return total_loss / total_num, total_dist / total_length

def split_dataset(wav_paths, script_paths, valid_ratio=0.05):
    train_loader_count = config.workers # 얘는 뭐하는 친구야?
    records_num = len(wav_paths) # wav_paths는 wav 파일의 개수 ex)29805

    batch_num = math.ceil(records_num / config.batch_size) # 올림 ex) 994
    valid_batch_num = math.ceil(batch_num * valid_ratio) # 올림 ex) 50
    train_batch_num = batch_num - valid_batch_num # ex) 944

    batch_num_per_train_loader = math.ceil(train_batch_num / config.workers) # ex) 236

    train_begin = 0
    train_end_raw_id = 0
    train_dataset_list = list() # <- 여기에 append 시킴

    for i in range(config.workers): # ex) 4번 반복

        train_end = min(train_begin + batch_num_per_train_loader, train_batch_num) # 둘중에 작은거 선택
        train_begin_raw_id = train_begin * config.batch_size # batch 사이즈의 시작을 구한다.
        train_end_raw_id = train_end * config.batch_size # batch 사이즈의 end를 구한다.

        train_dataset_list.append(BaseDataset(
                                        wav_paths[train_begin_raw_id:train_end_raw_id], # 잘려진 wav_paths
                                        script_paths[train_begin_raw_id:train_end_raw_id], # 잘려진 script_paths
                                        SOS_token, EOS_token)) # basedataset을 붙여넣는다
        train_begin = train_end 
    
    #valid_dataset - 모델 성능 평가하기 위한 데이터 set
    valid_dataset = BaseDataset(wav_paths[train_end_raw_id:], script_paths[train_end_raw_id:], SOS_token, EOS_token)
    
    return train_batch_num, train_dataset_list, valid_dataset

def main():

    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    char2index, index2char = label_loader.load_label('./hackathon.labels') # 한글을 다 라벨링 해놈
    # label_loader 파일에 load_label를 실행 char2index 랑 index2char를 정리
    SOS_token = char2index['<s>']
    EOS_token = char2index['</s>']
    PAD_token = char2index['_']

    random.seed(config.seed) # 난수를 발생하기 위해서 적절한 시드를 난수 발생기에 주어야 한다.
    torch.manual_seed(config.seed) # cpu 연산 무작위 고정
    torch.cuda.manual_seed_all(config.seed) # 멀티 gpu 연산 무작위 고정

    have_cuda = torch.cuda.is_available()
    # gpu를 사용하려면면 텐서를 gpu 연산이 가능한 자료형으로 변환 해야한다.
    device = torch.device('cuda' if have_cuda else 'cpu')

   # cpu랑 gpu 둘 중에 어떤걸 쓸지 선택

    # N_FFT: defined in loader.py
    feature_size = N_FFT / 2 + 1 # N_FFT는 stft 윈도우 사이즈 -> 이거 아니지 않아?
 #----------------------------------------모델 생성 ----------------------------------------------------------------------#
    # encoder
    enc = EncoderRNN(feature_size, config.hidden_size,
                     input_dropout_p=config.dropout, dropout_p=config.dropout,
                     n_layers=config.layer_size, bidirectional=config.bidirectional, rnn_cell='gru', variable_lengths=False)

    # decoder
    dec = DecoderRNN(len(char2index), config.max_len, config.hidden_size * (2 if config.bidirectional else 1),
                     SOS_token, EOS_token,
                     n_layers=config.layer_size, rnn_cell='gru', bidirectional=config.bidirectional,
                     input_dropout_p=config.dropout, dropout_p=config.dropout, use_attention=config.use_attention)

    model = Seq2seq(enc, dec) # seq2seq 모델에 enc, dec 삽입
    model.flatten_parameters() # data parallel 을 위해서 사용 (gpu를 사용)

    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08) # 왜 -0.08 ~ 0.08 로 해줬지?
    
    model = model.to(device)
    model = nn.DataParallel(model).to(device) # GPU를 사용하기 위해서  scatter 한다. 병렬처리를 위해

    optimizer = optim.Adam(model.module.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_token).to(device)

    if config.mode != "train": # train 모드가 아닐때는 여기서 끝냄
        return
 
 #-------------------------------------------데이터 분할-------------------------------------------------------------------#
    data_list = config.data_csv_path
    #data_list.csv'
    wav_paths = list() # ex) sample_dataset/train/train_data/41_0601_211_0_07930_02.wav 들이 모여있음
    script_paths = list() # ex) ./sample_dataset/train/train_data/41_0601_211_0_07930_02.label 들이 모여있음

    with open(data_list, 'r') as f: # 읽기 모드로 datalist를 연다. 
        for line in f:
            wav_path, script_path = line.strip().split(',')
            wav_paths.append(os.path.join(config.DATASET_PATH, 'train_data', wav_path))
            script_paths.append(os.path.join(config.DATASET_PATH, 'train_data', script_path))

    best_loss = 1e10
    begin_epoch = 0

    # load all target scripts for reducing disk i/o
    target_path = os.path.join(config.DATASET_PATH, 'train_label')

    load_targets(target_path) # loader.py의 load_targets실행
    # 요걸로 key, target이 정해짐
    # 41_0508_171_0_08412_03,566 610 304 509 251 662 748 528 662 519 662 749 62 661 123 662
    # load_targets을 실행함으로써 target_dict에 위에 정보가 들어감
    # 정답 기록한 것

    train_batch_num, train_dataset_list, valid_dataset = split_dataset(wav_paths, script_paths, valid_ratio=0.05)
    #train_batch_num, train_dataset_list, valid_dataset으로 split_dataset한다.
    #train data에서 모델의 성능을 평가하기 위해서 validation set을 분리한다.
    
    logger.info('start') # 루트 로거에 수준 INFO 메시지를 로그 합니다. 인자는 debug()처럼 해석

    train_begin = time.time() # 현재 시간
 #---------------------------------------------학습 시작---------------------------------------------------------------------------#
    for epoch in range(begin_epoch, config.max_epochs): # epoch이 몇번 돌지 0 ~ 10

        # 여기서 큐에 정확히 뭐들어가는지 확인인
        train_queue = queue.Queue(config.workers * 2) # defualt = 4
        # FIFO 큐의 생성자. ( A ) -> 큐에 배치할 수 있는 항목 수에 대한 상한을 설정하는 정수

        train_loader = MultiLoader(train_dataset_list, train_queue, config.batch_size, config.workers) # basedataloader를 이어 붙였음
        # train_queue = FIFO 큐 생성자, batch_size = 32 , workers = 4 
        train_loader.start() # MultiLoader 에서 start 함수 실행

        train_loss, train_cer = train(model, train_batch_num, train_queue, criterion, optimizer,
                                      device, train_begin, config.workers, 10, config.teacher_forcing)
        # model = nn.DataParallel(model).to(device) , criterion = nn.CrossEntropyLoss, optimizer = optim.Adam
        # train_cer = total_dist / total_length
        logger.info('Epoch %d (Training) Loss %0.4f CER %0.4f' % (epoch, train_loss, train_cer))

        train_loader.join() # MultiLoader 에서 join 함수 실행 이건 또 뭐하는 함수야? -> 쓰레드가 완료될 때까지 기다린다 (정확하게 어떤말인지 ...)

        valid_queue = queue.Queue(config.workers * 2)
        valid_loader = BaseDataLoader(valid_dataset, valid_queue, config.batch_size, 0)
        valid_loader.start() # BaseDataLoader 에서 start 함수 실행

        eval_loss, eval_cer = evaluate(model, valid_loader, valid_queue, criterion, device)
        logger.info('Epoch %d (Evaluate) Loss %0.4f CER %0.4f' % (epoch, eval_loss, eval_cer))

        valid_loader.join() # BaseDataLoader 에서 join 함수 실행  쓰레드가 완료될 때까지 기다린다

        best_model = (eval_loss < best_loss)

        if best_model:
            if not os.path.isdir("weight"):
                os.makedirs("weight")
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print("파일을 저장했어요")
            torch.save(state, os.path.join('./weight/model.pt'))

            best_loss = eval_loss

 #------------------------------------------------ model save ------------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
