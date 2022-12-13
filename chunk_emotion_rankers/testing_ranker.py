#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:40:37 2019

@author: winston
"""
import torch
import sys
import numpy as np
import pandas as pd
from utils import getPaths, DiffRslChunkSplitTestingData_Ranker
from tqdm import tqdm
import torch.nn.functional as F
from model import LSTMnet_ranker
from scipy.io import loadmat
from scipy.stats import mode, spearmanr, kendalltau
import argparse
# Fix random seeds
seed = 31
np.random.seed(seed)



argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epochs", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
args = vars(argparse.parse_args())

# Parameters
epochs = int(args['epochs'])
batch_size = int(args['batch_size'])
emo_attr = args['emo_attr']
rdn_ratio = 0.1
resample = 10

# # Parameters
# epochs = 20
# batch_size = 128
# emo_attr = 'Dom'
# rdn_ratio = 0.1
# resample = 10

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

# Loading Norm-Feature
Feat_mean_All = loadmat('../NormTerm/feat_norm_means.mat')['normal_para']
Feat_std_All = loadmat('../NormTerm/feat_norm_stds.mat')['normal_para'] 

# too many pairs need to predict => random downsample sample & repeat 10 times
SpearCorr_ResampleRsl = []
Ktau_ResampleRsl = []
for times in range(resample):
    _paths, _labels = getPaths(label_csv, split_set='Test', emo_attr=emo_attr)
    rdn_idx = np.random.choice(len(_paths), int(len(_paths)*rdn_ratio), replace=False)
    _paths, _labels = _paths[rdn_idx], _labels[rdn_idx]

    # generate ground-truth ranking
    table = pd.concat([pd.Series(_paths, name='FileName'), pd.Series(_labels, name='EmoLabel')],axis=1)
    table['Rank'] = table['EmoLabel'].rank(method ='min', ascending=False) 

    # prediction process
    File_Name = []
    Prefer_Score = []
    for idx1 in tqdm(range(len(_paths)), file=sys.stdout):
        # Loading Data1 & Normalization
        data1 = loadmat(root_dir + _paths[idx1].replace('.wav','.mat'))['Audio_data'] 
        data1 = data1[:,1:]  # remove time-info
        data1 = (data1-Feat_mean_All)/Feat_std_All
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data1[np.isnan(data1)]=0
        data1[data1>3]=3
        data1[data1<-3]=-3        
        # Preference Prediction process
        prefer_score = []
        for idx2 in range(len(_paths)):
            if _paths[idx1]==_paths[idx2]:
                pass
            else:
                # Loading Data2 & Normalization
                data2 = loadmat(root_dir + _paths[idx2].replace('.wav','.mat'))['Audio_data'] 
                data2 = data2[:,1:]  # remove time-info
                data2 = (data2-Feat_mean_All)/Feat_std_All
                # Bounded NormFeat Range -3~3 and assign NaN to 0
                data2[np.isnan(data2)]=0
                data2[data2>3]=3
                data2[data2<-3]=-3            
                # Data Processing 
                chunk_data1, chunk_data2 = DiffRslChunkSplitTestingData_Ranker([data1],[data2])
                chunk_data1, chunk_data2 = torch.from_numpy(chunk_data1), torch.from_numpy(chunk_data2)
                chunk_data1, chunk_data2 = chunk_data1.cuda(), chunk_data2.cuda()
                chunk_data1, chunk_data2 = chunk_data1.float(), chunk_data2.float()     
                # Model prediction
                pred_rsl = model(chunk_data1, chunk_data2)
                # output predictions for f1-score calculation
                pred_prob = F.softmax(pred_rsl)
                pred_prob = pred_prob.data.cpu().numpy()
                pred_class = np.argmax(pred_prob,axis=1)
                major_class = mode(pred_class)[0][0]        # major vote prediction clas 
                prefer_score.append(major_class)
        prefer_score = sum(prefer_score)
        File_Name.append(str(_paths[idx1]))
        Prefer_Score.append(prefer_score)
    File_Name = np.array(File_Name)
    Prefer_Score = np.array(Prefer_Score)        

    # generate predicted ranking
    pred_table = pd.concat([pd.Series(File_Name, name='FileName'), pd.Series(Prefer_Score, name='EmoPreferScore')],axis=1)
    pred_table['PredRank'] = pred_table['EmoPreferScore'].rank(method ='min', ascending=False) 

    # compute ranking performance
    merge_table = table.merge(pred_table, how = 'inner', on = ['FileName'])
    corr = spearmanr(merge_table['PredRank'].values, merge_table['Rank'].values)[0]
    kend = kendalltau(merge_table['PredRank'].values, merge_table['Rank'].values)[0]
    SpearCorr_ResampleRsl.append(corr)
    Ktau_ResampleRsl.append(kend)
    
print('Emotion: '+emo_attr)
print('Random Ratio: '+str(rdn_ratio))
print('Re-sample Times: '+str(resample))
print('Number of Test Sentences: '+str(len(merge_table)))
print('Spearman Corr.: '+str(np.mean(SpearCorr_ResampleRsl)))
print('Kendall’s tau: '+str(np.mean(Ktau_ResampleRsl)))

