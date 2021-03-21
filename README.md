# SCL-Indonesian-Address-Element-Extraction
Shopee Code League 2021 Data Science Competition

Only supports up to Tensor Flow 1.14 and Python 3.7. For GPU accleration, Nvidia CUDA 10.0 and cuDNN 7.2 is required.
Training time is about 30min, prediction time is about 17min. Tested on AMD 3700X with RTX 2060 Super.

Credit to Guillaume Genthial (https://guillaumegenthial.github.io) for original code and idea. This repo is modified based on the chars-lstm-lstm-crf model.

Model Arch:

1. GloVe 840B vectors
2. Chars embeddings
3. Chars Bi-LSTM with dropout = 0.125
4. Bi-LSTM with dropout = 0.125
5. CRF (Conditional Random Field) using IOBES tagging scheme

Related Paper Neural Architectures for Named Entity Recognition by Lample et al.

Achieved 0.40125 score on Kaggle private leaderboard, F1 of 0.8713461, precision of 0.8530295, recall of 0.8904665, final loss of 0.5650089

Usage:
1. data cleaning.py
2. data prep.py
3. build_vocab.py
4. build_glove.py (download and put glove glove.840B.300d.txt in root folder)
5. Main.py (modified DATADIR)
6. export.py (modified DATADIR)
7. serve.py
