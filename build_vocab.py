"""Script to build words, chars and tags vocab"""

__author__ = "Original script by Guillaume Genthial, modified by Billy Cao 2021"

import sys
from collections import Counter
from pathlib import Path

print(f'Running on Python {sys.version}.')

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 1

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    def words(name):
        return f'{name}.words.txt'


    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'val']:
        with Path(words(n)).open(encoding='utf-8') as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path('vocab.words.txt').open('w', encoding='utf-8') as f:
        for w in sorted(list(vocab_words)):
            f.write(f'{w}\n')
    print(f'- done. Kept {len(vocab_words)} out of {len(counter_words)}')

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        vocab_chars.update(w)

    with Path('vocab.chars.txt').open('w', encoding="ascii", errors='ignore') as f:
        counter_chars = 0
        for c in sorted(list(vocab_chars)):
            if c.isascii():
                f.write(f'{c}\n')
                counter_chars += 1
            else:
                continue
        f.write("/\n")  # Manually add the missed out slash
        # f.write("?\n")  # Manually add the question mark that was filtered out
    skipped = len(vocab_chars) - counter_chars
    print(f'- done. Found {counter_chars} chars, skipped {skipped} non-ASCII characters.')

    # 3. Tags
    # Get all tags from the training set


    def tags(name):
        return f'{name}.tags.txt'


    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(tags('train')).open(encoding='utf-8') as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path('vocab.tags.txt').open('w', encoding='utf-8') as f:
        for t in sorted(list(vocab_tags)):
            f.write(f'{t}\n')
    print(f'- done. Found {len(vocab_tags)} tags.')
