#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:09:05 2019

@author: winstonlin
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
import pandas as pd



class RankerEmoDataset(Dataset):

    def __init__(self, root_dir, split_set, emo_attr):
        # Parameters
        self.root_dir = root_dir
        self.emo_attr = emo_attr
        self.split_set = split_set
        # Loading Paths & Labels
        if split_set=='Train':
            _table = pd.read_csv('./Rank_Labels/training_pairs_'+emo_attr+'.csv')
        elif split_set=='Validation':
            _table = pd.read_csv('./Rank_Labels/validation_pairs_'+emo_attr+'.csv')
        self.sentence1 = _table['Sentence1'].values.astype('str')
        self.sentence2 = _table['Sentence2'].values.astype('str')
        self.prefer_label = _table['PreferLabel'].values.astype('int')
        # Loading Norm-Feature
        self.Feat_mean_All = loadmat('../NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std_All = loadmat('../NormTerm/feat_norm_stds.mat')['normal_para']  

    def __len__(self):
        if self.split_set == 'Train':
            return len(pd.read_csv('./Rank_Labels/training_pairs_'+self.emo_attr+'.csv'))
        elif self.split_set == 'Validation':
            return len(pd.read_csv('./Rank_Labels/validation_pairs_'+self.emo_attr+'.csv'))
        
    def __getitem__(self, idx):
        # Loading Data1 & Normalization
        data1 = loadmat(self.root_dir + self.sentence1[idx].replace('.wav','.mat'))['Audio_data'] 
        data1 = data1[:,1:]  # remove time-info
        data1 = (data1-self.Feat_mean_All)/self.Feat_std_All
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data1[np.isnan(data1)]=0
        data1[data1>3]=3
        data1[data1<-3]=-3  
        
        # Loading Data2 & Normalization
        data2 = loadmat(self.root_dir + self.sentence2[idx].replace('.wav','.mat'))['Audio_data'] 
        data2 = data2[:,1:]  # remove time-info
        data2 = (data2-self.Feat_mean_All)/self.Feat_std_All
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data2[np.isnan(data2)]=0
        data2[data2>3]=3
        data2[data2<-3]=-3          
        
        # Loading Label
        label = self.prefer_label[idx]
        return data1, data2, label
