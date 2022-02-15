#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:44:00 2019

@author: winston
"""
import pandas as pd
import numpy as np
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

# Split Original batch Data into Small-Chunk batch Data Structure with different step window size
def DiffRslChunkSplitTrainingData_Ranker(Batch_data1, Batch_data2, Batch_label):
    """
    Note!!! This function can't process sequence length which less than given chunk_size (i.e.,1sec=62frames)
    Please make sure all your input data's length are greater then given chunk_size
    """
    chunk_size = 62  # (62-frames*0.016sec) = 0.992sec
    chunk_num = 11
    n = 2
    num_shifts = n*chunk_num-1
    Split_Data1 = []
    Split_Label = np.array([])
    for i in range(len(Batch_data1)):
        data = Batch_data1[i]
        label = Batch_label[i]
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
            Split_Data1.append( data[start_idx[iii]: end_idx[iii]] )  
        split_label = np.repeat( label,len(start_idx) )    
        Split_Label = np.concatenate((Split_Label,split_label))
    Split_Data2 = []
    for i in range(len(Batch_data2)):
        data = Batch_data2[i]
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
            Split_Data2.append( data[start_idx[iii]: end_idx[iii]] )        
    return np.array(Split_Data1), np.array(Split_Data2), Split_Label

# Split Original batch Data into Small-Chunk batch Data Structure with different step window size
def DiffRslChunkSplitTestingData_Ranker(Batch_data1, Batch_data2):
    """
    Note!!! This function can't process sequence length which less than given chunk_size (i.e.,1sec=62frames)
    Please make sure all your input data's length are greater then given chunk_size
    """
    chunk_size = 62  # (62-frames*0.016sec) = 0.992sec
    chunk_num = 11
    n = 2
    num_shifts = n*chunk_num-1
    Split_Data1 = []
    for i in range(len(Batch_data1)):
        data = Batch_data1[i]
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
            Split_Data1.append( data[start_idx[iii]: end_idx[iii]] )   
    Split_Data2 = []
    for i in range(len(Batch_data2)):
        data = Batch_data2[i]
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
            Split_Data2.append( data[start_idx[iii]: end_idx[iii]] )
    return np.array(Split_Data1), np.array(Split_Data2)

# Split Original batch Data into Small-Chunk batch Data Structure with different step window size
def DiffRslChunkSplitTestingData(Batch_data):
    """
    Note!!! This function can't process sequence length which less than given chunk_size (i.e.,1sec=62frames)
    Please make sure all your input data's length are greater then given chunk_size
    """
    chunk_size = 62  # (62-frames*0.016sec) = 0.992sec
    chunk_num = 11
    n = 2
    num_shifts = n*chunk_num-1
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
