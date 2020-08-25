import pickle
import torch
from torch.utils.data import Dataset

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
    
    
class Bert_Blacklist_dataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        sample = self.data[index]
        instance = {
            'id': sample['ID'],
            'words': sample['token_id'],
            'tag' : sample['label'],
            'segment': [0]*len(sample['token_id']),
            'mask': [1]*(len(sample['token_id']))
        }
        return instance
    
    def collate_fn(self, samples):
        batch = {}
        for key in ['id']:
            if any(key not in sample for sample in samples):
                continue
            batch[key] = [sample[key] for sample in samples]
            
        for key in ['words', 'tag', 'segment', 'mask']:
            if any(key not in sample for sample in samples):
                continue
            to_len = max([len(sample[key]) for sample in samples])
            if key =='tag':
                pad_logit = 2
            else:
                pad_logit = 0
            padded = pad_to_len(
                [sample[key] for sample in samples], 512, pad_logit
            )
            batch[key] = torch.tensor(padded).long()

        return batch



