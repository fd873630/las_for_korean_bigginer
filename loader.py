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

import os
import sys
import math
import wavio
import time
import torch
import random
import threading
import logging
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger('root') # 로거 이름이 패키지/모듈 계층을 추적한다는 것을 의미하며, 로거 이름으로부터 이벤트가 기록되는 위치를 직관적으로 명확히 알 수 있다.
FORMAT = "[%(asctime)s %(filename)s:%(lineno)s - %(funcName)s()] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=FORMAT) # sys.stdout(표준출력장치,모니터))
logger.setLevel(logging.INFO)

# logging. 프로그램의 정상 작동 중에 발생하는 이벤트 보고 (가령 상태 모니터링이나 결함 조사)
# 로깅이란 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단

PAD = 0
N_FFT = 512
SAMPLE_RATE = 16000

target_dict = dict()

def load_targets(path):
    with open(path, 'r') as f:
        for no, line in enumerate(f):
            key, target = line.strip().split(',')
            target_dict[key] = target
    # 41_0508_171_0_08412_03,566 610 304 509 251 662 748 528 662 519 662 749 62 661 123 662
    # 이걸 , 기준으로 나눔

def get_spectrogram_feature(filepath): # file을 spectrogram을 사용
    (rate, width, sig) = wavio.readwav(filepath)
    #sig = (1000, 1) 이렇게 나옴
    sig = sig.ravel() # 위에 걸 (1000,) 이렇게 만듬

    stft = torch.stft(torch.FloatTensor(sig),
                        N_FFT,
                        hop_length=int(0.01*SAMPLE_RATE),
                        win_length=int(0.030*SAMPLE_RATE),
                        window=torch.hamming_window(int(0.030*SAMPLE_RATE)),
                        center=False,
                        normalized=False,
                        onesided=True)
    # (* \times N \times T \times 2)
    # where :math:`*` is the optional batch size of :attr:`input`,
    # :math:`N` is the number of frequencies where STFT is applied, N은 STFT가 적용되는 주파수 수
    # :math:`T` is the total number of frames used, and each pair 프레임 사용되는 `T`는 총 번호, 각 쌍
    # in the last dimension represents a complex number as the real part and the imaginary part.
    stft = (stft[:,:,0].pow(2) + stft[:,:,1].pow(2)).pow(0.5); # stft (r^2 + i^2)^0.5
    amag = stft.numpy();
    feat = torch.FloatTensor(amag) # 가로축 시간(프레임) 세로축 사용된 frequency, 색
    feat = torch.FloatTensor(feat).transpose(0, 1) # 왜 돌린거지?

    return feat

# basedataset에서 start token, end token추가
def get_script(filepath, bos_id, eos_id):
    key = filepath.split('/')[-1].split('.')[0]
    script = target_dict[key]
    tokens = script.split(' ')
    result = list()
    result.append(bos_id)
    for i in range(len(tokens)):
        if len(tokens[i]) > 0:
            result.append(int(tokens[i]))
    result.append(eos_id)
    return result

class TestDataset(Dataset):
    '''

    직접 음성을 넣어서 잘 나오는지 확인하기 위한 클래스

    직접 만들었어연!

    '''
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        script = 0
        return feat, script

class BaseDataset(Dataset):
    def __init__(self, wav_paths, script_paths, bos_id=1307, eos_id=1308):
        self.wav_paths = wav_paths
        self.script_paths = script_paths
        self.bos_id, self.eos_id = bos_id, eos_id

    def __len__(self):
        return len(self.wav_paths)

    def count(self):
        return len(self.wav_paths)

    def getitem(self, idx):
        feat = get_spectrogram_feature(self.wav_paths[idx])
        script = get_script(self.script_paths[idx], self.bos_id, self.eos_id)

        return feat, script

def _collate_fn(batch):
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(PAD)
    #print(targets.shape)
    #print(targets.shape)
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

  
    return seqs, targets, seq_lengths, target_lengths

def _collate_fn_2(batch):
    def seq_length_(p):
        return len(p[0])

    #def target_length_(p):
    #    return len(p[1])

    seq_lengths = [len(s[0]) for s in batch]
    #target_lengths = [len(s[1]) for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    #max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    #max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    #targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    #targets.fill_(PAD)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)
        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        #targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    return seqs, seq_lengths

class BaseDataLoader(threading.Thread): # thread -> 파이썬 코드를 순차적으로 실행
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn = _collate_fn
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, targets, seq_lengths, target_lengths

    def run(self): # start() 실행하면 요게 실행됨
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size): 
                if self.index >= self.dataset_count:
                    break

                items.append(self.dataset.getitem(self.index))
                self.index += 1

            if len(items) == 0:
                batch = self.create_empty_batch()
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn(items)
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class TestDataLoader(threading.Thread): # thread -> 파이썬 코드를 순차적으로 실행
    def __init__(self, dataset, queue, batch_size, thread_id):
        threading.Thread.__init__(self)
        self.collate_fn_2 = _collate_fn_2
        self.dataset = dataset
        self.queue = queue
        self.index = 0
        self.batch_size = batch_size
        self.dataset_count = dataset.count()  # wav path 의 길이
        self.thread_id = thread_id

    def count(self):
        return math.ceil(self.dataset_count / self.batch_size)

    def create_empty_batch(self):
        seqs = torch.zeros(0, 0, 0)
        targets = torch.zeros(0, 0).to(torch.long)
        seq_lengths = list()
        target_lengths = list()
        return seqs, seq_lengths

    def run(self):  # start() 실행하면 요게 실행됨
        logger.debug('loader %d start' % (self.thread_id))
        while True:
            items = list()

            for i in range(self.batch_size):  # batch_size = 1 이면

                if self.index >= self.dataset_count:  # 인덱스 길이를 path의 길이만큼 늘린거
                    break

                items.append(self.dataset.getitem(self.index))  # index 0 부터 시작함 0부터 ~ 쭉 item에 append 시킴
                self.index += 1  # 인덱스 증가

            if len(items) == 0:  # 아이템 길이가 0일때
                batch = self.create_empty_batch()  # seqs , targets, seq_lengths, target_lengths 0으로 만들고
                self.queue.put(batch)
                break

            random.shuffle(items)

            batch = self.collate_fn_2(items)  # 여기에서 값을 집어 넣음
            self.queue.put(batch)
        logger.debug('loader %d stop' % (self.thread_id))

class MultiLoader(): 
    def __init__(self, dataset_list, queue, batch_size, worker_size):
        self.dataset_list = dataset_list
        self.queue = queue
        self.batch_size = batch_size
        self.worker_size = worker_size
        self.loader = list() # basedataloader를 이어 붙임

        for i in range(self.worker_size):
            self.loader.append(BaseDataLoader(self.dataset_list[i], self.queue, self.batch_size, i))

    def start(self):
        for i in range(self.worker_size):
            self.loader[i].start() # i 번째 로더 실행

    def join(self): # 이 쓰레드가 완료될 때까지 기다린다
        for i in range(self.worker_size):
            self.loader[i].join()

