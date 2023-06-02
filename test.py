import pickle
from typing import List
import json
from llama.tokenizer import Tokenizer
import tqdm
def pklSentences():
    sentences = []
    with open('./00.jsonl', 'r') as f:
        for i in tqdm.tqdm(range(200_000)):
            line = f.readline()
            text = json.loads(line)['text']
            sentences.append(text)
   
    pickle.dump(sentences, open('train_sentences.pkl', 'wb'))
    sentences = []
    with open('./val.jsonl', 'r') as f:
        for i in range(10_000):
            line = f.readline()
            text = json.loads(line)['text']
            sentences.append(text)
    pickle.dump(sentences, open('val_sentences.pkl', 'wb'))
def encodeSentences():
    vocab = Tokenizer(model_path='tokenizer.model')
    sentences = pickle.load(open('train_sentences.pkl', 'rb'))[:100_000]
    embed = [vocab.encode(s, bos=True, eos=True) for s in tqdm.tqdm(sentences)]
    pickle.dump(embed, open('train_embed_nonFlat.pkl', 'wb'))
    
    sentences = pickle.load(open('val_sentences.pkl', 'rb'))[:10]
    train_embed = [vocab.encode(s, bos=True, eos=True) for s in tqdm.tqdm(sentences)]
    pickle.dump(train_embed, open('val_embed_nonFlat.pkl', 'wb'))
def main():
    encodeSentences()

    
main()