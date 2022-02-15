#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:13:34 2020

@author: winston
"""
import os
from utils import parse_EmoDetails
import numpy as np
import csv
# Fix random seeds
seed = 31
np.random.seed(seed)


# QA-approach to define the preference label
def QA_table(sent1_label, sent2_label):
    equal = []
    greater = []
    lower = [] 
    for ii in range(len(sent1_label)):
        diff = sent1[ii]-np.array(sent2_label)
        equal.append( sum(diff==0) )
        greater.append( sum(diff>0) )
        lower.append( sum(diff<0) )
    equal = sum(equal)
    greater = sum(greater)
    lower = sum(lower)
    total = equal+greater+lower
    return np.array([greater/total, lower/total, equal/total])
###############################################################################


# Parameters
label_dir = '/media/winston/UTD-MSP/Speech_Datasets/MSP-PODCAST-Publish-1.6/Labels/labels_detailed.json'
split_set = 'Train'
emo_attr = 'Val'
threshold = 0.6
rdn_ratio = 0.1

# obtain annotations for each sentence w.r.t the desired emotions and sub-set
Fname, worker_rsl = parse_EmoDetails(label_dir, split_set=split_set, emo_attr=emo_attr)

# main process to determine the preference pairs
prefer_pairs = []
prefer_label = []
for i in range(len(Fname)):
    rdn_idx = np.random.choice(len(Fname), int(len(Fname)*rdn_ratio), replace=False)
    for j in range(len(rdn_idx)):
        name1, name2 = str(Fname[i]), str(Fname[rdn_idx[j]])
        sent1, sent2 = worker_rsl[i], worker_rsl[rdn_idx[j]]
        prefer_prob = QA_table(sent1, sent2)
        if np.max(prefer_prob)>=threshold:
            if np.argmax(prefer_prob)==0:
                prefer_label.append(1)
                prefer_pairs.append((name1, name2))
            elif np.argmax(prefer_prob)==1:
                prefer_label.append(0)
                prefer_pairs.append((name1, name2))
            else:
                pass
        else:
            pass

# creating saving repo
if not os.path.isdir('./Rank_Labels/'):
    os.makedirs('./Rank_Labels/')

# output CSV file for labels
if split_set=='Train':
    f = open('./Rank_Labels/training_pairs_'+emo_attr+'.csv','w')
elif split_set=='Validation':
    f = open('./Rank_Labels/validation_pairs_'+emo_attr+'.csv','w')
w = csv.writer(f)    
w.writerow(('Sentence1', 'Sentence2', 'PreferLabel')) 
for i in range(len(prefer_pairs)):
    sentence1, sentence2 = prefer_pairs[i][0], prefer_pairs[i][1]
    label = prefer_label[i]
    w.writerow((sentence1, sentence2, label))         
f.close()           

