from torch.utils.data import Dataset
from typing import List, Dict
import json
import numpy as np
import torch

class TextDataset(Dataset):
    def __init__(self, sequencified_data, tokens_per_seq: int):
        self.data = sequencified_data # an list of bpt x num_batches tokinized lists of variable length
        self.tokens_per_seq = tokens_per_seq
    def __len__(self):
        return (len(self.data)-1) // self.tokens_per_seq
    def __getitem__(self, idx):
        
        data, labels = get_batch(self.data, idx* self.tokens_per_seq, self.tokens_per_seq)
        return data, labels

def batchify(data: torch.Tensor, batch_size: int):
    """Creates a bach matrix of batch_size x num batches
    Args:
        data torch.tensor: flattened tokinzied data
        batch_size (int): size of each batch
    """
    num_batch = len(data) // batch_size
    data = data[0:num_batch*batch_size]
    data = data.view(batch_size, -1).T
    return data

def get_batch(batched_data:torch.Tensor, batch_num: int, bptt: int):
    seq_len = min(bptt, len(batched_data) - 1 - batch_num)
    data = batched_data[batch_num:batch_num+seq_len]
    target = batched_data[batch_num + 1 : batch_num + seq_len + 1].flatten()
    return data, target

def loadSentencesFromJson(path_to_json: str) -> List[str]:
    with open(path_to_json, 'r') as f:
        contents = f.read()
    # Split the contents into lines
    lines = contents.splitlines()

    sentences: List[str] = []
    for line in lines:
        record = json.loads(line)
        sentences.append(record['text'])
    return sentences