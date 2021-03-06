"""

Copyright 2017- IBM Corporation

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

import random

import numpy as np

import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .attention import Attention
from .baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        #input_var 정답 [1 ,20, 820]
        # self.forward_step(decoder_input, decoder_hidden, encoder_outputs,function=function)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(embedded, hidden)


        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)


        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)

        # predicted_softmax가 이제 decoder 라인바이 라인으로 820개의 결과값이
        # torch.Size([2, 1, 820])
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()
        
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio) # beam search 하려면 이거 고쳐야됨
     
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        beam_search = False # 지호 추가
        decoder_outputs = []
        sequence_symbols = []
        sequence_symbols_beam = []
        if beam_search:
            lengths = np.array([max_length] * batch_size)

            def decode_1(step, step_output, step_attn):  # 여기 만져야됨
                # decode(di, step_output, step_attn)

                # torch.Size([2, 820]) batch / 820개의 확률
                decoder_outputs.append(step_output)

                if self.use_attention:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

                symbols = decoder_outputs[-1].topk(1)[1]  # top 1 이였으면 하나만 나옴 지금 5로 올려놨음
                symbols_beam = decoder_outputs[-1].topk(5)[1]
                sequence_symbols.append(symbols)
                sequence_symbols_beam.append(symbols_beam)

                #print("sequence_symbols_beam")
                #print(sequence_symbols_beam)



                eos_batches = symbols.data.eq(self.eos_id)  # 이게 eos_id 감지

                if eos_batches.dim() > 0:
                    eos_batches = eos_batches.cpu().view(-1).numpy()
                    update_idx = ((lengths > step) & eos_batches) != 0
                    lengths[update_idx] = len(sequence_symbols)

                return symbols

            decoder_input = inputs[:, 0].unsqueeze(1)  # 초기값

            for di in range(max_length):

                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              function=function)

                # torch.Size([2, 820]) batch / 820개의 확률
                step_output = decoder_output.squeeze(1)

                # torch.Size([2, 820]) batch / 820개의 확률

                symbols = decode_1(di, step_output, step_attn)

                decoder_input = symbols  # 요거 였구만

        else:
            lengths = np.array([max_length] * batch_size)

            def decode(step, step_output, step_attn): # 여기 만져야됨
                #decode(di, step_output, step_attn)

                # torch.Size([2, 820]) batch / 820개의 확률
                decoder_outputs.append(step_output)

                if self.use_attention:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

                symbols = decoder_outputs[-1].topk(1)[1] # top 1 이였으면 하나만 나옴 지금 5로 올려놨음
                symbols_beam = decoder_outputs[-1].topk(5)[1]

                sequence_symbols.append(symbols)
                sequence_symbols_beam.append(symbols_beam)

                eos_batches = symbols.data.eq(self.eos_id) # 이게 eos_id 감지 추가해야됨

                if eos_batches.dim() > 0:
                    eos_batches = eos_batches.cpu().view(-1).numpy()

                    update_idx = ((lengths > step) & eos_batches) != 0

                    lengths[update_idx] = len(sequence_symbols)
                    #print(len(sequence_symbols))

                return symbols

            # Manual unrolling is used to support random teacher forcing.
            # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
            if use_teacher_forcing:
                
                decoder_input = inputs[:, :-1] # 정답 / 이거 왜함? 어짜피 길이 안맞으면 짧은건 819 eos 남아 있는데? 정답이고
                #print(inputs.shape)
                #print(inputs[:,:-1].shape)
                decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)

                # decoder_output.shpae [2 , 19 ,820] 배치, 추정 길이 , 각 길이마다 확률
                #print(decoder_output.shape)
                #print("decoder_output.shape")

                #print(decoder_output.size(1))
                for di in range(decoder_output.size(1)):
                    #di는 어팬드 된거임
                    step_output = decoder_output[:, di, :]
                    #print(step_output)
                    if attn is not None:
                        step_attn = attn[:, di, :]
                    else:
                        step_attn = None
                    decode(di, step_output, step_attn)

            else:
                decoder_input = inputs[:, 0].unsqueeze(1) # 정답
                
                for di in range(max_length):

                    decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                             function=function)
                    #step_outputs, hidden = self.forward_step(input, hidden, encoder_outputs)


                    step_output = decoder_output.squeeze(1)

                    # torch.Size([2, 820]) batch / 820개의 확률

                    symbols = decode(di, step_output, step_attn)

                    decoder_input = symbols #요거 였구만
                    # 0일때 5개



            ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
            ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        #batch_size 여기에서 배치사이즈

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length

    def _get_length_penalty(self, length, alpha=1.2, min_length=5):
        """
        Calculate length-penalty.
        because shorter sentence usually have bigger probability.
        using alpha = 1.2, min_length = 5 usually.
        """
        return ((min_length + length) / (min_length + 1)) ** alpha
