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
from utils import DiffRslChunkSplitTrainingData_Ranker
from torch import nn 
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dataloader import RankerEmoDataset
from torch.utils.data.sampler import SubsetRandomSampler
from model import LSTMnet_ranker
from sklearn.metrics import f1_score
import argparse



def collate_fn(batch):  
    data1, data2, label = zip(*batch)   
    chunk_data1, chunk_data2, chunk_label = DiffRslChunkSplitTrainingData_Ranker(data1, data2, label)
    return torch.from_numpy(chunk_data1), torch.from_numpy(chunk_data2), torch.from_numpy(chunk_label)
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
# epochs = 20
# batch_size = 128
# emo_attr = 'Act'
# shuffle = True

# LSTM-model loading
model = LSTMnet_ranker(input_dim=130, hidden_dim=130 , output_dim=2, num_layers=2)
model.cuda()

# settings
SAVING_PATH = './Models/'
root_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Features/OpenSmile_lld_IS13ComParE/feat_mat/'
training_dataset = RankerEmoDataset(root_dir, split_set='Train', emo_attr=emo_attr)
validation_dataset = RankerEmoDataset(root_dir, split_set='Validation', emo_attr=emo_attr)

# creating repo
if not os.path.isdir(SAVING_PATH):
    os.makedirs(SAVING_PATH)

# shuffle training dataset by generating random indices 
train_indices = list(range(len(training_dataset)))
valid_indices = list(range(len(validation_dataset)))

# create an optimizer for training
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# emotion-recog model training
Epoch_trainLoss_All = []
Epoch_trainF1score_All = []
Epoch_validLoss_All = []
Epoch_validF1score_All = []
val_accuracy = 0
for ep in range(epochs):
    # too many training/validation pairs => random sample to train/validate for each epoch
    train_sample_num = 100000
    valid_sample_num = 100000
    if shuffle :
        np.random.shuffle(train_indices)
        np.random.shuffle(valid_indices)
        
    # creating data samplers and loaders:
    select_train_indices = train_indices[:train_sample_num]
    train_sampler = SubsetRandomSampler(select_train_indices)
    train_loader = torch.utils.data.DataLoader(training_dataset, 
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=12,
                                               pin_memory=True,
                                               collate_fn=collate_fn)
    
    select_valid_indices = valid_indices[:valid_sample_num]
    valid_sampler = SubsetRandomSampler(select_valid_indices)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, 
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=12,
                                               pin_memory=True,
                                               collate_fn=collate_fn)    
    # training process
    print('training process')
    model.train()
    batch_loss_train_all = []
    Pred_train_Class = []
    True_train_Class = []    
    for i_batch, data_batch in enumerate(tqdm(train_loader, file=sys.stdout)):
        input_tensor1, input_tensor2, target_tensor = data_batch
        # Input Tensor Data1
        input_var1 = torch.autograd.Variable(input_tensor1.cuda())
        input_var1 = input_var1.float()
        # Input Tensor Data2
        input_var2 = torch.autograd.Variable(input_tensor2.cuda())
        input_var2 = input_var2.float()        
        # Input Tensor Targets
        target_var = torch.autograd.Variable(target_tensor.cuda())
        target_var = target_var.long()
        # models flow
        pred_rsl = model(input_var1, input_var2)  
        # BCE loss for binary classification
        loss = criterion(pred_rsl, target_var)
        # output predictions for f1-score calculation
        pred_prob = F.softmax(pred_rsl)
        pred_prob = pred_prob.data.cpu().numpy()
        pred_class = np.argmax(pred_prob,axis=1)
        true_class = target_tensor.data.cpu().numpy()
        Pred_train_Class.extend(list(pred_class))
        True_train_Class.extend(list(true_class))              
        # output loss for record
        batch_loss_train_all.append(loss.data.cpu().numpy())
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        # clear GPU memory
        torch.cuda.empty_cache()    
    Epoch_trainLoss_All.append(np.mean(batch_loss_train_all))
    fs_train = f1_score(True_train_Class, Pred_train_Class, average='macro')
    Epoch_trainF1score_All.append(fs_train)
    
    # validation process 
    print('validation process')
    model.eval()
    batch_loss_valid_all = []
    Pred_valid_Class = []
    True_valid_Class = []
    for i_batch, data_batch in enumerate(tqdm(valid_loader, file=sys.stdout)):  
        input_tensor1, input_tensor2, target_tensor = data_batch
        # Input Tensor Data1
        input_var1 = torch.autograd.Variable(input_tensor1.cuda())
        input_var1 = input_var1.float()
        # Input Tensor Data2
        input_var2 = torch.autograd.Variable(input_tensor2.cuda())
        input_var2 = input_var2.float() 
        # Input Tensor Targets
        target_var = torch.autograd.Variable(target_tensor.cuda())
        target_var = target_var.long()
        # models flow
        pred_rsl = model(input_var1, input_var2)  
        # BCE loss for binary classification
        loss = criterion(pred_rsl, target_var)
        # output predictions for f1-score calculation
        pred_prob = F.softmax(pred_rsl)
        pred_prob = pred_prob.data.cpu().numpy()
        pred_class = np.argmax(pred_prob,axis=1)
        true_class = target_tensor.data.cpu().numpy()
        Pred_valid_Class.extend(list(pred_class))
        True_valid_Class.extend(list(true_class))           
        # output loss for record
        batch_loss_valid_all.append(loss.data.cpu().numpy())        
        torch.cuda.empty_cache()
    Epoch_validLoss_All.append(np.mean(batch_loss_valid_all))
    fs_valid = f1_score(True_valid_Class, Pred_valid_Class, average='macro')
    Epoch_validF1score_All.append(fs_valid)    

    print('Epoch: '+str(ep)+' ,Training-f1score: '+str(fs_train)+' ,Validation-f1score: '+str(fs_valid))
    print('=================================================================')

    # Checkpoint for saving best model based on val-loss
    if ep==0:
        val_accuracy = fs_valid 
        torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+emo_attr+'_ranker.pth.tar'))
        print("=> Saving the initial best model (Epoch="+str(ep)+")")
    else:
        if val_accuracy < fs_valid:
            torch.save(model.state_dict(), os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+emo_attr+'_ranker.pth.tar'))
            print("=> Saving a new best model (Epoch="+str(ep)+")")
            print("=> Accuracy increasing from "+str(val_accuracy)+" to "+str(fs_valid) )
            val_accuracy = fs_valid
        else:
            print("=> Validation Accuracy did not improve (Epoch="+str(ep)+")")
    print('=================================================================')

# Drawing Loss Curve for Epoch-based and Batch-based
plt.title('Epoch-Loss Curve')
plt.plot(Epoch_trainLoss_All,color='blue',linewidth=3)
plt.plot(Epoch_validLoss_All,color='red',linewidth=3)
plt.savefig(os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+emo_attr+'_ranker_loss.png'))
#plt.show()
plt.title('Epoch-F1score Curve')
plt.plot(Epoch_trainF1score_All, 'blue', linewidth=3)
plt.plot(Epoch_validF1score_All, 'red', linewidth=3)
plt.savefig(os.path.join(SAVING_PATH, 'LSTM_epoch'+str(epochs)+'_batch'+str(batch_size)+'_'+emo_attr+'_ranker_f1score.png'))
#plt.show()

