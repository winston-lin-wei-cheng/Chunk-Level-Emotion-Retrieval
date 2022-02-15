#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:44:00 2019

@author: winston
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import torch
import json


def getPaths(path_label, split_set, emo_attr):
    """
    This function is for filtering data by different constraints of label
    Args:
        path_label$ (str): path of label.
        split_set$ (str): 'Train', 'Validation' or 'Test' are supported.
        emo_attr$ (str): 'Act', 'Dom' or 'Val'
    """
    label_table = pd.read_csv(path_label)
    whole_fnames = (label_table['FileName'].values).astype('str')
    split_sets = (label_table['Split_Set'].values).astype('str')
    emo_act = label_table['EmoAct'].values
    emo_dom = label_table['EmoDom'].values
    emo_val = label_table['EmoVal'].values
    _paths = []
    _label_act = []
    _label_dom = []
    _label_val = []
    for i in range(len(whole_fnames)):
        # Constrain with Split Sets      
        if split_sets[i]==split_set:
            # Constrain with Emotional Labels
            _paths.append(whole_fnames[i])
            _label_act.append(emo_act[i])
            _label_dom.append(emo_dom[i])
            _label_val.append(emo_val[i])
        else:
            pass
    if emo_attr == 'Act':
        return np.array(_paths), np.array(_label_act) 
    elif emo_attr == 'Dom':
        return np.array(_paths), np.array(_label_dom)
    elif emo_attr == 'Val':
        return np.array(_paths), np.array(_label_val)

def evaluation_metrics(true_value,predicted_value):
    corr_coeff = np.corrcoef(true_value,predicted_value)
    ccc = 2*predicted_value.std()*true_value.std()*corr_coeff[0,1]/(predicted_value.var() + true_value.var() + (predicted_value.mean() - true_value.mean())**2)
    return(ccc,corr_coeff)

def cc_coef(output, target):
    mu_y_true = torch.mean(target)
    mu_y_pred = torch.mean(output)                                                                                                                                                                                              
    return 1 - 2 * torch.mean((target - mu_y_true) * (output - mu_y_pred)) / (torch.var(target) + torch.var(output) + torch.mean((mu_y_pred - mu_y_true)**2))

def CombineListToMatrix(Data):
    length_all = []
    for i in range(len(Data)):
        length_all.append(len(Data[i])) 
    feat_num = len(Data[0].T)
    Data_All = np.zeros((sum(length_all),feat_num))
    idx = 0
    Idx = []
    for i in range(len(length_all)):
        idx = idx+length_all[i]
        Idx.append(idx)        
    for i in range(len(Idx)):
        if i==0:    
            start = 0
            end = Idx[i]
            Data_All[start:end]=Data[i]
        else:
            start = Idx[i-1]
            end = Idx[i]
            Data_All[start:end]=Data[i]
    return Data_All  

# Split Original batch Data into Small-Chunk batch Data Structure with different step window size
def DiffRslChunkSplitData(Batch_data):
    """
    Note!!! This function can't process sequence length which less than given chunk_size (i.e.,1sec=62frames)
    Please make sure all your input data's length are greater then given chunk_size
    """
    chunk_size = 62  # (62-frames*0.016sec) = 0.992sec
    chunk_num = 11
    n = 2            # scaling factor: 11*2=22 chunks per sentence
    num_shifts = n*chunk_num-1  # max_length = 11sec, chunk needs to shift 10 times to obtain total 11 chunks for each utterance
    Split_Data = []
    for i in range(len(Batch_data)):
        data = Batch_data[i]
        # Shifting-Window size varied by differenct length of input utterance => Different Resolution
        step_size = int(int(len(data)-chunk_size)/num_shifts)      
        # Calculate index of chunks
        start_idx = [0]
        end_idx = [chunk_size]
        for iii in range(num_shifts):
            start_idx.extend([start_idx[0] + (iii+1)*step_size])
            end_idx.extend([end_idx[0] + (iii+1)*step_size])    
        # Output Split Data/Label
        for iii in range(len(start_idx)):
            Split_Data.append( data[start_idx[iii]: end_idx[iii]] )     
    # sampled chunk index
    sample_rate = 1
    sample_idx = np.arange(0, len(Split_Data), sample_rate)
    return np.array(Split_Data)[sample_idx]

def parse_EmoDetails(path, split_set, emo_attr):
    # Emotion Worker details
    with open(path) as json_file:
        data_all = json.load(json_file)
    # Emotion Table for Different Set Split
    label_table = pd.read_csv('/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv')    
    whole_fnames = (label_table['FileName'].values).astype('str')
    whole_split = (label_table['Split_Set'].values).astype('str')    
    # Parsing Emotion detail for corresponding set
    File_Name = []
    Workers_Act = []
    Workers_Dom = []
    Workers_Val = []
    for i in range(len(whole_fnames)):
        if whole_split[i]==split_set:
            fname = whole_fnames[i]
            File_Name.append(fname)
            data_fname = data_all[fname]
            workers_Act = []
            workers_Dom = []
            workers_Val = []
            for worker in data_fname.keys():
                workers_Act.append(data_fname[worker]['EmoAct'])
                workers_Dom.append(data_fname[worker]['EmoDom'])
                workers_Val.append(data_fname[worker]['EmoVal'])
            Workers_Act.append(workers_Act)
            Workers_Dom.append(workers_Dom)  
            Workers_Val.append(workers_Val) 
    if emo_attr=='Act':
        return np.array(File_Name), Workers_Act
    elif emo_attr=='Dom':
        return np.array(File_Name), Workers_Dom
    elif emo_attr=='Val':    
        return np.array(File_Name), Workers_Val

def build_labelDist(path, split_set, emo_attr):
    File_Name, Workers_Emo = parse_EmoDetails(path, split_set, emo_attr)
    gaussian_parameters_emo = []
    for i in range(len(File_Name)):
        samples_act = np.array(Workers_Emo[i])
        mu_emo, std_emo = norm.fit(samples_act)
        gaussian_parameters_emo.append((mu_emo, std_emo))
    # Create Distribution Dictionary
    emo_dist_dict = {};
    for k in range(len(File_Name)):
        dist_emo = {'Parameter':gaussian_parameters_emo[k]}  
        emo_dist_dict[File_Name[k]] = dist_emo 
    return File_Name, emo_dist_dict              

def smooth_seq(label_seq):
    N=3 # moving average window size
    smooth_seq = np.convolve(label_seq, np.ones((N,))/N, mode='valid')
    smooth_seq = np.insert(smooth_seq, 0, label_seq[0])
    smooth_seq = np.append(smooth_seq, label_seq[-1])
    # downsample smooth sequence
    sample_rate = 1
    sample_idx = np.arange(0, len(smooth_seq), sample_rate)
    sample_seq = smooth_seq[sample_idx]   
    return sample_seq

def generate_chunk_EmoSeq(mu, std, rank_seq):
    if std!=0:
        # assign the emotion trend of chunk-seq. (based on preference ranking result)
        # for 22-chunks per sentence
        trend = 0
        chunk_trend_seq = [trend]
        for i in range(len(rank_seq)):
            try:
                if (rank_seq[i+1]-rank_seq[i])>0:
                    trend = trend - 1
                    chunk_trend_seq.append(trend)
                elif (rank_seq[i+1]-rank_seq[i])<0:
                    trend = trend + 1
                    chunk_trend_seq.append(trend)
                elif (rank_seq[i+1]-rank_seq[i])==0:
                    trend = trend
                    chunk_trend_seq.append(trend)
            except:
                pass
        # smooth emo seq
        chunk_trend_seq = np.array(chunk_trend_seq)
        chunk_trend_seq = smooth_seq(chunk_trend_seq)
        # re-scale to original mean/std
        mu2 = np.mean(chunk_trend_seq)
        std2 = np.std(chunk_trend_seq)
        chunk_trend_seq = mu + (chunk_trend_seq-mu2)*(std/std2)
    else:
        chunk_trend_seq = np.array([mu]*len(rank_seq))
    return chunk_trend_seq

