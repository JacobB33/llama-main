import sentencepiece as spm
from typing import List, Dict
import json

import argparse

parser = argparse.ArgumentParser(description='Specify the name of the dataset you want to parse, and trains a \
                                 sentincepiece processor on it')
parser.add_argument('file_path', type=str, help='the name of the file, ie val (not val.jsonl)')
parser.add_argument('--load_to_text_file', action='store_true', help='load to text file first with the same name')
parser.add_argument('--vocab_size', type=int, default=1000)
parser.add_argument('--num_threads', type=int, default=16)
parser.add_argument('--max_sentence_len', type=int, default=4192)
parser.add_argument('--save_str', type=str, default='Model')

args = parser.parse_args()

def jsonlToTextFile(file_path: str):
    """
    Reads a JSONL file and extracts the 'text' field from each record.
    Writes the extracted text to a new file in plain text format.

    Args:
        file_path (str): The path to the JSONL file to be read.

    Returns:
        None
    """
    with open(file_path, 'r') as f:
        contents = f.read()
    # Split the contents into lines
    lines = contents.splitlines()

    data: List[Dict] = []
    for line in lines:
        record = json.loads(line)
        data.append(record)

    with open('val.txt', 'w') as f:
        for i in range(len(data)):
            f.write(data[i]['text'] + '\n')

   
def main():
    if args.load_to_text_file:
        jsonlToTextFile(args.file_path + '.jsonl')
    input_file = args.file_path + '.txt'
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=f'{args.file_path}{args.save_str}',
        vocab_size=args.vocab_size,
        pad_id=0,                
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[CLS]',
        eos_piece='[SEP]',
        user_defined_symbols='[MASK]',
        model_type='bpe',
        num_threads=args.num_threads
    )


if __name__ == "__main__":
    main()


