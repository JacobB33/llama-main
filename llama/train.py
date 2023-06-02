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
parser = argparse.ArgumentParser(description='Train code for transformer')
parser.add_argument('tokinizer_path', type=str, help='path to tokeinizer')
parser.add_argument('data_path', type=str, help='path to data.json file')
parser.add_argument('save_path', type=str, help='Path to folder to save results')
parser.add_argument('--seq_len', type=int, default=800, help='The amount of tokens per sequence')
parser.add_argument('--batch_size', type=int, default=16, help='the batch size')
parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--clip', type=float, default=1, help='what to clip the grad by')

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def eval(model: nn.Module, test_loader: DataLoader, vocab_len: int, epoch: int):
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        for i, (data, labels) in tqdm.tqdm(enumerate(test_loader)):
            data, labels = data.to(device), labels.to(device)
            preds = model(data, 0)
            loss = criterion(preds.reshape(-1, vocab_len), labels.flatten())
            
            loss.backward(retain_graph=True)            
            losses += loss.detach().item()
        losses /= len(test_loader)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, vocab_len: int):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    total_losses = []
    huuuge_losses = []
    epoch_validation = []

    for epoch in range(args.num_epochs):
        losses = 0
        print(f'starting epoch {epoch}')
        for i, (data, labels) in tqdm.tqdm(enumerate(train_loader)):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data, 0)
            loss = criterion(preds.reshape(-1, vocab_len), labels.flatten())
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            losses += loss.item()
            huuuge_losses.append(loss.item())
            if(i % 100 == 0):
                print(loss.item())
        losses /= len(train_loader)
        eval_loss = eval(model, val_loader, vocab_len, epoch)
        print(f'For epoch {epoch}, there was an average loss of {losses} and val of {eval_loss}')
        total_losses.append(losses)
        torch.save(model.state_dict(), f'{args.save_path}/model_epoch_{epoch}.pt')
    
            

def main():
    modelArgs = ModelArgs()
    modelArgs.n_layers = 4
    modelArgs.n_heads = 4
    modelArgs.dim = 128
    modelArgs.max_seq_len = args.seq_len
    modelArgs.max_batch_size = args.batch_size
    
    vocab = Tokenizer(model_path=args.tokinizer_path)
    modelArgs.vocab_size = vocab.n_words
    
    train_loader, val_loader = getTrainAndValDataLoaders(args.batch_size, args.seq_len)
    print(f'post train loader = {torch.cuda.memory_allocated()}')
    model = Transformer(modelArgs)
    print(f'post model = {torch.cuda.memory_allocated()}')
    train(model, train_loader, val_loader, modelArgs.vocab_size)
    print('done')

if __name__ == "__main__":
    main()