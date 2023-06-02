import itertools
import pickle

import tqdm
from tokenizer import Tokenizer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import List, Dict
import json
import numpy as np
import torch


class EncodedSentenceDataset(Dataset):
    def __init__(self, flat_data: torch.Tensor, sequence_len: int):
        self.data = flat_data # .contiguous()
        self.seq_len = sequence_len
    def __len__(self):
        return (len(self.data) -1) // self.seq_len
    def __getitem__(self, idx):
        data = self.data[idx * self.seq_len: (idx + 1) * self.seq_len]
        targets = self.data[idx * self.seq_len + 1: (idx + 1) * self.seq_len + 1]
        return data, targets

def getTrainAndValDataLoaders(batch_size:int,  sequence_length: int, paralel: bool = False):
    """This method needs to have the two pkl files of the
    encoded data already on your computer. To do this, you have to download
    and extract the 00.jsonl file from the pile inside of your llama-main directory
    Then after that, run createTestValData.py and then you can cal this function.
    """
    train_encodings = pickle.load(open('./train_embed_nonFlat.pkl', 'rb'))
    val_encodings = pickle.load(open('./val_embed_nonFlat.pkl', 'rb'))
    
    train_encodings = torch.tensor(list(itertools.chain.from_iterable(train_encodings)))
    val_encodings = torch.tensor(list(itertools.chain.from_iterable(val_encodings)))

    train_ds = EncodedSentenceDataset(train_encodings, sequence_length)
    test_ds = EncodedSentenceDataset(val_encodings, sequence_length)
    
    if not paralel:
        train_ds = DataLoader(train_ds, batch_size, shuffle=False,)
        test_ds = DataLoader(test_ds, batch_size, shuffle=False)
    else:
        train_ds =  DataLoader(
            train_ds, batch_size=batch_size, pin_memory=True, 
            shuffle=False, sampler=DistributedSampler(train_ds))
        test_ds =  DataLoader(
            test_ds, batch_size=batch_size, pin_memory=True, 
            shuffle=False, sampler=DistributedSampler(test_ds))
    return train_ds, test_ds


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
    train_ds = EncodedSentenceDataset(flattened_train, seq_len)
    if not parallel:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
        return DataLoader(
            train_ds, batch_size=batch_size, pin_memory=True, 
            shuffle=False, sampler=DistributedSampler(train_ds))