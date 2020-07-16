#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import pandas as pd
import pickle
import os
import pickle

from opencc import OpenCC

import torch
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer

pretrained_bert = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(pretrained_bert, do_lower_case=True)
current_path = os.getcwd()


cc = OpenCC('s2t')  # convert from Simplified Chinese to Traditional Chinese
# can also set conversion by calling set_conversion
cc.set_conversion('s2tw')



def pad_to_len(seqs, to_len, padding=0):
    paddeds = []
    for seq in seqs:
        paddeds.append(
            seq[:to_len] + [padding] * max(0, to_len - len(seq))
        )

    return paddeds

class Bert_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'words':[101] + sample['words'],
            'tag' : [0] + sample['tag'],
            'segment': [0]*len(sample['words']),
            'mask': [1] * (len(sample['words'])+1)
        }
        return instance
    
    def collate_fn(self, samples):
        batch = {}
        for key in ['words', 'tag', 'segment', 'mask']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            if key =='tag':
                pad_logit = 2
            else:
                pad_logit = 0
            padded = pad_to_len(
                [sample[key] for sample in samples], 128, pad_logit
            )
            batch[key] = torch.tensor(padded)

        return batch

    
    
    
    

with open(f'{current_path}/module/pkl/boston.pkl', 'rb')as f:
    train = pickle.load(f)

with open(f'{current_path}/module/pkl/asia_institute.pkl', 'rb')as f:
    train2 = pickle.load(f)

with open(f'{current_path}/module/pkl/people.pkl', 'rb')as f:
    train3 = pickle.load(f)

all_train = train + train2 + train3

dataset = Bert_dataset(all_train)

with open(f'{current_path}/module/pkl/bert_input.pkl', 'wb')as f:
    pickle.dump(dataset, f)



