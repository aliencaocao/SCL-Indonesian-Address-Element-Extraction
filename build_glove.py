"""Build an np.array from some glove file and some vocab file
You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

__author__ = "Original script by Guillaume Genthial, modified by Billy Cao 2021"

import sys
from pathlib import Path

import numpy as np

print(f'Running on Python {sys.version}.')

if __name__ == '__main__':
    # Load vocab
    with Path('vocab.words.txt').open(encoding='utf-8') as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path('glove.840B.300d.txt').open(encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                percentage = round(line_idx / 2100000, 2) * 100
                print(f'- At line {line_idx}/2100000 ({percentage}%)')
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print(f'- done. Found {found} vectors for {size_vocab} words')

    # Save np.array to file
    np.savez_compressed('glove.npz', embeddings=embeddings)
