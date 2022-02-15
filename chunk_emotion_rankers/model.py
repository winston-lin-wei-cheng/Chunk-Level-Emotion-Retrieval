#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:07:21 2019

@author: winstonlin
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class LSTMnet_ranker(torch.nn.Module): 
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMnet_ranker, self).__init__()
        # Net Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        # shared LSTM-layers
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=0.5, batch_first=True, bidirectional=False)
        # Dense-Output-layers
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.output_dim) 
              
    def forward(self, inputs1, inputs2):
        # LSTM-info flow
        lstm_out1, _ = self.lstm(inputs1) 
        # FC output1
        outputs1 = self.fc1(lstm_out1[:,-1,:])
        outputs1 = F.relu(outputs1)
        outputs1 = self.fc2(outputs1)
        # LSTM-info flow
        lstm_out2, _ = self.lstm(inputs2) 
        # FC output2
        outputs2 = self.fc1(lstm_out2[:,-1,:])
        outputs2 = F.relu(outputs2)
        outputs2 = self.fc2(outputs2)        
        out = outputs1-outputs2
        return out

