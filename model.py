#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:07:21 2019

@author: winstonlin
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np  


class LSTMnet(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # Dense-Output-layers(Seq)
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
              
    def forward(self, inputs):
        # LSTM-info flow
        lstm_out, lstm_hidden = self.lstm(inputs) 
        # Seq/Mean label output
        outputs = self.fc1(lstm_out[:,-1,:])
        outputs = F.relu(outputs)
        outputs = self.fc2(outputs)
        outputs = outputs.squeeze(1)
        outputs_mean = []
        # we use 22-chunks per sentence   
        for i_batch in np.arange(0,len(outputs),22):
            outputs_mean.append(torch.mean(outputs[i_batch:i_batch+22]))             
        outputs_mean = torch.stack(outputs_mean)
        return outputs, outputs_mean  

