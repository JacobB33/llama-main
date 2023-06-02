import os
import pickle
import tqdm
from nonParalelModel import *
import argparse
from tokenizer import Tokenizer
from trainutils import *
import itertools
from torch.utils.data import DataLoader
import torch
import fire
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Train code for transformer')
parser.add_argument('tokinizer_path', type=str, help='path to tokeinizer')
parser.add_argument('save_path', type=str, help='Path to folder to save results')
parser.add_argument('--seq_len', type=int, default=800, help='The amount of tokens per sequence')
parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--clip', type=float, default=1, help='what to clip the grad by')

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
import time
start_time = time.time()    

def eval(model: nn.Module, test_loader: DataLoader, vocab_len: int, epoch: int):
    print(f'running an eval episode {epoch}')
    losses = 0
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        for i, (data, labels) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            data, labels = data.to(device), labels.to(device)
            preds = model(data, 0)
            loss = criterion(preds.reshape(-1, vocab_len), labels.flatten())
            losses += loss.item()
        losses /= len(test_loader)
    return losses
def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, vocab_len: int):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    total_losses = []
    huuuge_losses = []
    epoch_validation = []

    for epoch in range(args.num_epochs):
        losses = 0
        print(f'starting epoch {epoch}')
        for i, (data, labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data, 0)
            loss = criterion(preds.reshape(-1, vocab_len), labels.flatten())
            
            loss.backward()
            optimizer.step()
            
            losses += loss.item()
            huuuge_losses.append(loss.item())
            if(i % 1000 == 0):
                print(loss.item())
        losses /= len(train_loader)
        eval_loss = eval(model, val_loader, vocab_len, epoch)
        print(f'For epoch {epoch}, there was an average loss of {losses} and val of {eval_loss}')
        total_losses.append(losses)
        epoch_validation.append(eval_loss)
        epoch_save_path = args.save_path + f'/epoch_{epoch}/'
        if not os.path.exists(epoch_save_path):
            os.makedirs(epoch_save_path)
        torch.save(model.state_dict(), f'{epoch_save_path}/model_epoch_{epoch}.pt')
        plt.cla(); plt.clf()
        plt.plot(total_losses, label='train_loss', color='blue')
        plt.plot(epoch_validation, label='val_loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{epoch_save_path}/losses.png')
        print(f'Training on a 4090 took {time.time() - start_time} seconds')

    # Create a plot using plt of the total_losses and teh epoch_validation and save it to args.save_path
    plt.cla(); plt.clf()
    plt.plot(total_losses, label='train_loss', color='blue')
    plt.plot(epoch_validation, label='val_loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{args.save_path}/final_losses.png')
    
    
    
            

def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    elif not os.path.isdir(args.save_path):
        raise ValueError(f"{args.save_path} is not a directory")

    modelArgs = ModelArgs()
    modelArgs.n_layers = 3
    modelArgs.n_heads = 4
    modelArgs.dim = 128
    modelArgs.max_seq_len = args.seq_len
    modelArgs.max_batch_size = args.batch_size
    
    vocab = Tokenizer(model_path=args.tokinizer_path)
    modelArgs.vocab_size = vocab.n_words
    print('loading train and val loaders')
    train_loader, val_loader = getTrainAndValDataLoaders(args.batch_size, args.seq_len)
    model = Transformer(modelArgs)
    print("Starting Training!")
    train(model, train_loader, val_loader, modelArgs.vocab_size)
    print(f'Training on a 4090 took {time.time() - start_time} seconds')
    print('amazing 🎉')




if __name__ == "__main__":
    main()