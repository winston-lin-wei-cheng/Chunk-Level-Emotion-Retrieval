#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:09:05 2019

@author: winstonlin
"""
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat
from utils import build_labelDist, getPaths, generate_chunk_EmoSeq
import json



class MspPodcastDataset_training(Dataset):
    """MSP-Podcast Emotion dataset."""

    def __init__(self, root_dir, label_dir, rank_dir, emo_attr):
        # Parameters
        self.root_dir = root_dir
        self.label_dir = label_dir
        # Loading Label Distribution
        self._paths, self._labels_dist = build_labelDist(label_dir, split_set='Train', emo_attr=emo_attr)
        # Loading Label Ranking Sequence
        with open(rank_dir) as json_file:
            self.seq_rank_all = json.load(json_file)
        # Loading Norm-Feature
        self.Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0] 
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]  
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0] 

    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data & Normalization
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        data = data[:,1:]  # remove time-info
        data = (data-self.Feat_mean_All)/self.Feat_std_All
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Loading Label & Ranking Sequence
        mu, std = self._labels_dist[self._paths[idx]]['Parameter']
        rankSeq = self.seq_rank_all[self._paths[idx]]['ChunkRankSeq']
        # generate chunk-emo-sequence (EmoRankerSeq)
        label_seq = generate_chunk_EmoSeq(mu, std, rankSeq)        
        # Label Normalization
        label_seq = (label_seq-self.Label_mean)/self.Label_std
        label = (mu-self.Label_mean)/self.Label_std
        return data, label_seq, label

class MspPodcastDataset_validation(Dataset):
    """MSP-Podcast Emotion dataset."""

    def __init__(self, root_dir, label_dir, emo_attr):
        # Parameters
        self.root_dir = root_dir
        self.label_dir = label_dir
        # Loading Paths
        self._paths, self._labels = getPaths(label_dir, split_set='Validation', emo_attr=emo_attr) 
        # Loading Norm-Feature
        self.Feat_mean_All = loadmat('./NormTerm/feat_norm_means.mat')['normal_para']
        self.Feat_std_All = loadmat('./NormTerm/feat_norm_stds.mat')['normal_para']  
        # Loading Norm-Label
        if emo_attr == 'Act':
            self.Label_mean = loadmat('./NormTerm/act_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/act_norm_stds.mat')['normal_para'][0][0] 
        elif emo_attr == 'Dom':
            self.Label_mean = loadmat('./NormTerm/dom_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/dom_norm_stds.mat')['normal_para'][0][0]  
        elif emo_attr == 'Val':
            self.Label_mean = loadmat('./NormTerm/val_norm_means.mat')['normal_para'][0][0]
            self.Label_std = loadmat('./NormTerm/val_norm_stds.mat')['normal_para'][0][0]  

    def __len__(self):
        return len(self._paths)
    
    def __getitem__(self, idx):
        # Loading Data & Normalization
        data = loadmat(self.root_dir + self._paths[idx].replace('.wav','.mat'))['Audio_data'] 
        data = data[:,1:]  # remove time-info
        data = (data-self.Feat_mean_All)/self.Feat_std_All
        # Bounded NormFeat Range -3~3 and assign NaN to 0
        data[np.isnan(data)]=0
        data[data>3]=3
        data[data<-3]=-3        
        # Label Normalization
        label = (self._labels[idx]-self.Label_mean)/self.Label_std
        return data, label

