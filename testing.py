#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:54:49 2019

@author: winston
"""
from utils import getPaths, evaluation_metrics, DiffRslChunkSplitData
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import torch
from model import LSTMnet
import argparse



# argparse = argparse.ArgumentParser()
# argparse.add_argument("-ep", "--epochs", required=True)
# argparse.add_argument("-batch", "--batch_size", required=True)
# argparse.add_argument("-emo", "--emo_attr", required=True)
# args = vars(argparse.parse_args())

# # Parameters
# epochs = int(args['epochs'])
# batch_size = int(args['batch_size'])
# emo_attr = args['emo_attr']

# Parameters
epochs = 30
emo_attr = 'Val'
batch_size = 128

# Data/Label Dir
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']
if emo_attr == 'Act':
    Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Dom':    
    Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]
elif emo_attr == 'Val': 
    Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
    Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]    

# Regression Task => Prediction & De-Normalize Target
MODEL_PATH = './trained_seq2seq_model_v1.6/LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_EmoRankerSeq_'+emo_attr+'.pth.tar'
model = LSTMnet(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH))
model.cuda()
model.eval()

# Regression Task
test_file_path, test_file_tar = getPaths(label_dir, split_set='Test', emo_attr=emo_attr)
# test_file_path, test_file_tar = getPaths(label_dir, split_set='Validation', emo_attr=emo_attr)

# Testing Data & Label
Test_Pred = []
Test_Label = []
for i in tqdm(range(len(test_file_path))):
    data = loadmat(root_dir + test_file_path[i].replace('.wav','.mat'))['Audio_data']
    data = data[:,1:]                           # remove time-info
    data = (data-Feat_mean_All)/Feat_std_All    # Feature Normalization
    # Bounded NormFeat Range -3~3 and assign NaN to 0
    data[np.isnan(data)]=0
    data[data>3]=3
    data[data<-3]=-3
    # Model Prediction
    chunk_data = DiffRslChunkSplitData([data])
    chunk_data = torch.from_numpy(chunk_data)
    chunk_data = chunk_data.cuda()
    chunk_data = chunk_data.float() 
    # for directly mean target CCC
    _, pred_rsl = model(chunk_data)
    pred_rsl = pred_rsl.data.cpu().numpy()
    # Output prediction results
    Test_Pred.append(pred_rsl)
    Test_Label.append(test_file_tar[i])
Test_Pred = np.array(Test_Pred).reshape(-1)
Test_Label = np.array(Test_Label)

# Regression Task => Prediction & De-Normalize Target
Test_Pred = (Label_std*Test_Pred)+Label_mean

# Output Predict Reulst
pred_Rsl_CCC = str(evaluation_metrics(Test_Label, Test_Pred)[0])
print('EmoRankerSeq_Info')
print('Epochs: '+str(epochs))
print('Batch_Size: '+str(batch_size))
print('Model_Type: LSTM')
print(emo_attr+'-CCC: '+str(pred_Rsl_CCC))

