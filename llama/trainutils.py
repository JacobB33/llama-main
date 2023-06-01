import itertools

import tqdm
from tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
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

class DumbDataset(Dataset):
    def __init__(self, flat_data: torch.Tensor, sequence_len: int):
        self.data = flat_data # .contiguous()
        self.seq_len = sequence_len
    def __len__(self):
        return (len(self.data) -1) // self.seq_len
    def __getitem__(self, idx):
        data = self.data[idx * self.seq_len: (idx + 1) * self.seq_len]
        targets = self.data[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]
        return data, targets

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


def getTrainDataLoader(vocab: Tokenizer, data_path: str, batch_size: int, seq_len: int, parallel=False):
    print('loading json')
    train_sentences = loadSentencesFromJson(data_path)[:1000]
    print('encoding sentences')
    train_encodings = itertools.chain.from_iterable([vocab.encode(s, bos=True, eos=True) for s in tqdm.tqdm(train_sentences)])
    print('creating dataset')
    flattened_train = torch.tensor(list(train_encodings)).flatten()
    print(len(flattened_train))
    # batched_train = batchify(flattened_train, args.seq_len)
    # train_ds = TextDataset(batched_train, args.bptt)
    train_ds = DumbDataset(flattened_train, seq_len)
    if not parallel:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
        return DataLoader(
            train_ds, batch_size=batch_size, pin_memory=True, 
            shuffle=False, sampler=DistributedSampler(train_ds))