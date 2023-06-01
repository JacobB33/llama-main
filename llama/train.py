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
parser.add_argument('--seq_len', type=int, default=10000, help='The amount of tokens per batch')
parser.add_argument('--num_epochs', type=int, default=6, help='number of epochs to train for')
parser.add_argument('--bptt', type=int, default=100, help='tokens per sequence trained on from each batch')
parser.add_argument('--clip', type=float, default=1, help='what to clip the grad by')

args = parser.parse_args()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


def getTrainDataLoader(vocab: Tokenizer):
    print('loading json')
    train_sentences = loadSentencesFromJson(args.data_path)[:5000]
    print('encoding sentences')
    train_encodings = itertools.chain.from_iterable([vocab.encode(s, bos=True, eos=True) for s in tqdm.tqdm(train_sentences)])
    print('creating dataset')
    flattened_train = torch.tensor(list(train_encodings)).flatten()
    print(len(flattened_train))
    batched_train = batchify(flattened_train, args.seq_len)
    train_ds = TextDataset(batched_train, args.bptt)
    print('dumping dataset')
    pickle.dump(train_ds, open('./val_batched_and_tokened.pkl', 'wb'))
    return DataLoader(train_ds, batch_size=1, shuffle=False)

def train(model: nn.Module, train_loader: DataLoader, vocab_len: int):
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    total_losses = []
    for epoch in range(args.num_epochs):
        losses = 0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.squeeze(0).to(device), labels.squeeze(0).to(device)
            model.zero_grad()
            preds = model(data, 0)
            loss = criterion(preds.reshape(-1, vocab_len), labels)
            # loss = torch.autograd.Variable(loss, requires_grad=True)
            loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            losses += loss.item()
            if(i % 20 == 0):
                print(loss.item())
        losses /= len(train_loader)
        print(f'For epoch {epoch}, there was an average loss of {losses}')
        total_losses.append(losses)
        torch.save(model.state_dict(), f'./saves/model_epoch_{epoch}.pt')
    
            

def main():
    modelArgs = ModelArgs()
    vocab = Tokenizer(model_path=args.tokinizer_path)
    modelArgs.vocab_size = vocab.n_words
    modelArgs.max_seq_len = 1024
    train_loader = getTrainDataLoader(vocab)
    model = Transformer(modelArgs)
    train(model, train_loader, modelArgs.vocab_size)
    print('done')

if __name__ == "__main__":
    # fire.Fire(main)
    main()