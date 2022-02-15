#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 15:40:37 2019

@author: winston
"""
import torch
import sys
import numpy as np
import os
from utils import cc_coef, DiffRslChunkSplitData
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from dataloader import MspPodcastDataset_training, MspPodcastDataset_validation
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMnet
import argparse



def collate_fn_train(batch):  
    data, label_seq, label = zip(*batch)   
    chunk_data = DiffRslChunkSplitData(data)
    label_seq = np.array(label_seq).reshape(-1)
    label = np.array(label)
    return torch.from_numpy(chunk_data), torch.from_numpy(label_seq), torch.from_numpy(label)

def collate_fn_valid(batch):  
    data, label = zip(*batch)   
    chunk_data = DiffRslChunkSplitData(data)
    label = np.array(label)
    return torch.from_numpy(chunk_data), torch.from_numpy(label)
###############################################################################


argparse = argparse.ArgumentParser()
argparse.add_argument("-ep", "--epochs", required=True)
argparse.add_argument("-batch", "--batch_size", required=True)
argparse.add_argument("-emo", "--emo_attr", required=True)
args = vars(argparse.parse_args())

# Parameters
epochs = int(args['epochs'])
batch_size = int(args['batch_size'])
emo_attr = args['emo_attr']
shuffle = True

# # Parameters
# epochs = 30
# emo_attr = 'Val'
# batch_size = 128
# shuffle = True

# LSTM-model loading
model = LSTMnet(input_dim=130, hidden_dim=130 , output_dim=1, num_layers=2)
model.cuda()

# PATH settings
SAVING_PATH = './Models/'
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_detailed.json'
label_csv = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_concensus.csv'
rank_dir = './chunk_emotion_rankers/EmoChunkRankSeq/training_'+emo_attr+'.json'

# creating saving repo
if not os.path.isdir(SAVING_PATH):
    os.makedirs(SAVING_PATH)

# loading datasets
training_dataset = MspPodcastDataset_training(root_dir, label_dir, rank_dir, emo_attr)
validation_dataset = MspPodcastDataset_validation(root_dir, label_csv, emo_attr)

# shuffle datasets by generating random indices 
train_indices = list(range(len(training_dataset)))
valid_indices = list(range(len(validation_dataset)))
if shuffle:
    np.random.shuffle(train_indices)

# creating data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
train_loader = torch.utils.data.DataLoader(training_dataset, 
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=12,
                                           pin_memory=True,
                                           collate_fn=collate_fn_train)

valid_sampler = SubsetRandomSampler(valid_indices)
valid_loader = torch.utils.data.DataLoader(validation_dataset, 
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=12,
                                           pin_memory=True,
                                           collate_fn=collate_fn_valid)

# create an optimizer for training
optimizer = optim.Adam(model.parameters(), lr=0.001)

# emotion-recog model training
Epoch_trainLoss_All = []
Epoch_validLoss_All = []
val_loss = 0
for ep in range(epochs):
    print('training process')
    model.train()
    batch_loss_train_all = []
    for i_batch, data_batch in enumerate(tqdm(train_loader, file=sys.stdout)):
        input_tensor, input_seq_target, input_mean_target = data_batch
        # Input Tensor Data
        input_var = torch.autograd.Variable(input_tensor.cuda())
        input_var = input_var.float()
        # Input Tensor Seq/Mean Targets
        seq_tar = torch.autograd.Variable(input_seq_target.cuda())
        mean_tar = torch.autograd.Variable(input_mean_target.cuda())       
        seq_tar = seq_tar.float()
        mean_tar = mean_tar.float()
        # models flow
        pred_seq, pred_mean = model(input_var)         
        # CCC loss for mean target
        loss_mean = cc_coef(pred_mean, mean_tar)
        # CCC loss for seq target 
        loss_seq = cc_coef(pred_seq, seq_tar)
        # Total-loss = CCC + CCC
        alpha = 0.5
        loss = alpha*loss_mean + (1-alpha)*loss_seq                
        batch_loss_train_all.append(loss.data.cpu().numpy())
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        # clear GPU memory
        torch.cuda.empty_cache()        
    Epoch_trainLoss_All.append(np.mean(batch_loss_train_all))
    
    print('validation process')
    model.eval()
    batch_loss_valid_all = []
    for i_batch, data_batch in enumerate(tqdm(valid_loader, file=sys.stdout)):  
        input_tensor, input_target = data_batch 
        # Input Tensor Data
        input_var = torch.autograd.Variable(input_tensor.cuda())
        input_var = input_var.float()
        # Input Tensor Mean Targets
        mean_tar = torch.autograd.Variable(input_target.cuda())         
        mean_tar = mean_tar.float()
        # models flow
        _, pred_mean = model(input_var)
        # loss calculation (only mean label CCC for validation)
        loss = cc_coef(pred_mean, mean_tar)
        batch_loss_valid_all.append(loss.data.cpu().numpy())  
        torch.cuda.empty_cache()
    Epoch_validLoss_All.append(np.mean(batch_loss_valid_all))
    print('Epoch: '+str(ep)+' ,Training-loss: '+str(np.mean(batch_loss_train_all))+' ,Validation-loss: '+str(np.mean(batch_loss_valid_all)))
    print('=================================================================')

    # Checkpoint for saving best model based on val-loss
    if ep==0:
        val_loss = np.mean(batch_loss_valid_all) 
        torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_EmoRankerSeq_'+emo_attr+'.pth.tar'))
        print("=> Saving the initial best model (Epoch="+str(ep)+")")
    else:
        if val_loss > np.mean(batch_loss_valid_all):
            torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_EmoRankerSeq_'+emo_attr+'.pth.tar'))
            print("=> Saving a new best model (Epoch="+str(ep)+")")
            print("=> Loss reduction from "+str(val_loss)+" to "+str(np.mean(batch_loss_valid_all)) )
            val_loss = np.mean(batch_loss_valid_all)
        else:
            print("=> Validation Loss did not improve (Epoch="+str(ep)+")")
    print('=================================================================')

# Drawing Loss Curve for Epochs
plt.title('Epoch-Loss Curve')
plt.plot(Epoch_trainLoss_All,color='blue',linewidth=3)
plt.plot(Epoch_validLoss_All,color='red',linewidth=3)
plt.savefig(os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_EmoRankerSeq_'+emo_attr+'.png'))
#plt.show()
