#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:25:37 2020

@author: winston
"""
import os
import torch
import sys
import numpy as np
import pandas as pd
from utils import getPaths, DiffRslChunkSplitTestingData
from tqdm import tqdm
import torch.nn.functional as F
from model import LSTMnet_ranker
from scipy.io import loadmat
import json
import argparse



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epochs", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
argparse.add_argument("-set", "--split_set", required=True)
args = vars(argparse.parse_args())

# Parameters
epochs = int(args['epochs'])
batch_size = int(args['batch_size'])
emo_attr = args['emo_attr']
split_set = args['split_set']

# # Parameters
# epochs = 20
# batch_size = 128
# emo_attr = 'Act'
# split_set = 'Train'

# LSTM-model loading
MODEL_PATH = './trained_ranker_model_v1.6/LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+emo_attr+'_ranker.pth.tar'
model = LSTMnet_ranker(input_dim=130, hidden_dim=130 , output_dim=2, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda()
model.eval()

# settings
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_detailed.json'
label_csv = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'
_paths, _labels = getPaths(label_csv, split_set=split_set, emo_attr=emo_attr)

# Loading Norm-Feature
Feat_mean_All = loadmat('../NormTerm/feat_norm_means.mat')['normal_para']
Feat_std_All = loadmat('../NormTerm/feat_norm_stds.mat')['normal_para'] 

# predicting preference over chunks within sentence
File_Name = []
Ranking_Sequence = []
for idx in tqdm(range(len(_paths)), file=sys.stdout):
    # Loading Data1 & Normalization
    data = loadmat(root_dir + _paths[idx].replace('.wav','.mat'))['Audio_data'] 
    data = data[:,1:]  # remove time-info
    data = (data-Feat_mean_All)/Feat_std_All
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3    
    # chunking process
    chunk_data = DiffRslChunkSplitTestingData([data])
    # prefer score over chunks
    prefer_score = []
    for i in range(len(chunk_data)):
        score = []
        for j in range(len(chunk_data)):
            if i==j:
                pass
            else:
                chunk1 = chunk_data[i]
                chunk1 = chunk1.reshape(1, chunk1.shape[0], chunk1.shape[1])
                chunk2 = chunk_data[j]
                chunk2 = chunk2.reshape(1, chunk2.shape[0], chunk2.shape[1])
                chunk1, chunk2 = torch.from_numpy(chunk1), torch.from_numpy(chunk2)
                chunk1, chunk2 = chunk1.cuda(), chunk2.cuda()
                chunk1, chunk2 = chunk1.float(), chunk2.float()  
                # prediction process
                pred_rsl = model(chunk1, chunk2)
                pred_prob = F.softmax(pred_rsl)
                pred_prob = pred_prob.data.cpu().numpy()
                pred_class = np.argmax(pred_prob,axis=1)[0]
                score.append(pred_class)
        prefer_score.append(sum(score))
    prefer_score = pd.DataFrame(prefer_score, columns=['EmoPreferScore'])
    prefer_score['PredRank'] = prefer_score['EmoPreferScore'].rank(method ='min', ascending=False) 
    Ranking_Sequence.append( prefer_score['PredRank'].values.astype('int').tolist() )
    File_Name.append(str(_paths[idx]))

# creating saving repo
if not os.path.isdir('./EmoChunkRankSeq/'):
    os.makedirs('./EmoChunkRankSeq/')

# Create Emotion Ranking Seqence Dictionary
emo_rank_dict = {};
for i in range(len(File_Name)):
    emo_rank = {'ChunkRankSeq':Ranking_Sequence[i]}  
    emo_rank_dict[File_Name[i]] = emo_rank 

if split_set=='Train':
    with open('./EmoChunkRankSeq/training_'+emo_attr+'.json','w') as f:
        json.dump(emo_rank_dict, f, indent=4)
elif split_set=='Validation':
    with open('./EmoChunkRankSeq/validation_'+emo_attr+'.json','w') as f:
        json.dump(emo_rank_dict, f, indent=4) 
elif split_set=='Test':
    with open('./EmoChunkRankSeq/testing_'+emo_attr+'.json','w') as f:
        json.dump(emo_rank_dict, f, indent=4) 

