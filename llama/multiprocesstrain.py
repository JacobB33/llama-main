# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import itertools
from typing import Tuple
import os
import sys
import torch.nn as nn
import tqdm
from trainutils import *
import torch
import fire
import time
import json
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from torch.utils.data import DataLoader
from model import *
from tokenizer import *
def ddp_setup():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

class Trainer:
    def __init__(self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        vocab_len: int
   ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.epochs_run = 0
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.vocab_len = vocab_len
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        preds = self.model(source, 0)
        loss = F.cross_entropy(preds.reshape(-1, self.vocab_len), targets.flatten())
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.local_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")  
    
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)      
    
def main(
    tokenizer_path: str,
    data_path: str,
    sequence_length: float = 512,
    batch_size: float = 16,
    num_epochs: int = 10,
):
    ddp_setup()
    modelArgs = ModelArgs()
    modelArgs.n_layers = 8
    modelArgs.max_batch_size = batch_size
    vocab = Tokenizer(model_path=tokenizer_path)
    modelArgs.vocab_size = vocab.n_words
    modelArgs.max_seq_len = 1024
    train_loader = getTrainDataLoader(vocab, data_path, batch_size, sequence_length, True)
    model = Transformer(modelArgs)
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model, train_loader, optimizer, 1, './snapshots/snapshot.pt', modelArgs.vocab_size)
    trainer.train(num_epochs)
    destroy_process_group()
    print('done')
    





if __name__ == "__main__":
    fire.Fire(main)
