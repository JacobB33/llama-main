from model import *
import argparse
from tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='Train code for transformer')
parser.add_argument('tokinizer_path', type=str, help='path to tokeinizer')
parser.add_argument('data_path', type=str, help='path to data.txt file')

args = parser.parse_args()


def getTrainDataLoader(vocab: Tokenizer):
    with open(args.data_path, 'r') as f:
        text = f.read()
    data = vocab.encode(text)


def main():
    modelArgs = ModelArgs
    model = Transformer(ModelArgs)
    vocab = Tokenizer(model_path=args.tokinizer_path)
    dataloader = getTrainDataLoader(vocab)


if __name__ == "__main__":
    main()