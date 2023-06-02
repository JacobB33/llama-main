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
        print(f'Eval at epoch {epoch} had loss {losses}')

def getTrainDataLoader(vocab: Tokenizer):
    print('loading json')
    train_sentences = loadSentencesFromJson(args.data_path)[:1000]
    print('encoding sentences')
    train_encodings = itertools.chain.from_iterable([vocab.encode(s, bos=True, eos=True) for s in tqdm.tqdm(train_sentences)])
    print('creating dataset')
    flattened_train = torch.tensor(list(train_encodings)).flatten()
    print(len(flattened_train))
    # batched_train = batchify(flattened_train, args.seq_len)
    # train_ds = TextDataset(batched_train, args.bptt)
    train_ds = DumbDataset(flattened_train, args.seq_len)
    print('dumping dataset')
    pickle.dump(train_ds, open('./val_batched_and_tokened.pkl', 'wb'))
    return DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)

def train(model: nn.Module, train_loader: DataLoader, vocab_len: int):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    total_losses = []
    huuuge_losses = []
    epoch_validation = []
    print(f'start of train = {torch.cuda.memory_allocated()}')

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
        print(f'For epoch {epoch}, there was an average loss of {losses}')
        total_losses.append(losses)
        torch.save(model.state_dict(), f'./saves/model_epoch_{epoch}.pt')
    
            

def main():
    modelArgs = ModelArgs()
    modelArgs.n_layers = 1
    modelArgs.n_heads = 1
    dim = 128
    
    modelArgs.max_batch_size = args.batch_size
    vocab = Tokenizer(model_path=args.tokinizer_path)
    modelArgs.vocab_size = vocab.n_words
    modelArgs.max_seq_len = 1024
    print(f'pre train loader = {torch.cuda.memory_allocated()}')
    train_loader = getTrainDataLoader(vocab)
    print(f'post train loader = {torch.cuda.memory_allocated()}')
    model = Transformer(modelArgs)
    print(f'post model = {torch.cuda.memory_allocated()}')
    train(model, train_loader, modelArgs.vocab_size)
    print('done')

if __name__ == "__main__":
    # fire.Fire(main)
    main()