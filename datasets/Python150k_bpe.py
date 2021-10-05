import os
import torch
import torch.utils.data as data
import numpy as np

from utils.utils import get_files, read_csv
from preprocessing.preprocessing import load_sentence_piece_vocabulary, get_bpe_model

UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'


def read_vocab(path_to_model):
    bpe_model = get_bpe_model(path_to_model)
    bpe_vocab = load_sentence_piece_vocabulary(bpe_model)

    return bpe_vocab[0], bpe_vocab[1]


def add_padding_token(vocab):
    return np.append(vocab, PADDING_TOKEN)


class Python150k_bpe(data.Dataset):
    """Python150k dataset loader.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    - context_size (``int``): Number of tokens before and after the target token
    """

    ast_train_dir = 'training'
    ast_val_dir = 'validation'
    ast_test_dir = 'test'
    vocab_file = 'voc.npy'

    spm_model = 'voc_bpe.model'

    def __init__(self,
                 root_dir,
                 mode='train',
                 lookback_tokens=1000,
                 chunk_size=10000, max_len_label=20):
        self.root_dir = root_dir
        self.mode = mode
        self.idx_cached = -1
        self.chunk_size = chunk_size
        self.lookback_tokens = lookback_tokens
        self.max_len_label = max_len_label

        self.vocab, self.word2idx, self.idx2word = self.init_vocab(os.path.join(root_dir, self.spm_model))

        self.vocab = add_padding_token(self.vocab)

        self.padding_idx = self.get_vocab_len() - 1

        if self.mode.lower() == 'train':
            # Get the training data and labels filepaths
            self.files = get_files(os.path.join(root_dir, self.ast_train_dir))

        elif self.mode.lower() == 'val':
            # Get the validation data and labels filepaths
            self.files = get_files(os.path.join(root_dir, self.ast_val_dir))

        elif self.mode.lower() == 'test':
            # Get the test data and labels filepaths
            self.files = get_files(os.path.join(root_dir, self.ast_test_dir))

        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def init_vocab(self, model):
        word2idx, idx2word = read_vocab(model)
        vocab = list(word2idx.keys())

        return vocab, word2idx, idx2word

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        """

        # Calculate the batch file
        file_idx = np.floor(index/self.chunk_size)

        # Load and cache data if cache missed
        if self.idx_cached != file_idx:
            self.data_cached = read_csv(self.files[file_idx])
            self.idx_cached = file_idx
            print("Cache miss")

        row = int(index - (self.chunk_size * file_idx))
        data_len = self.data_cached[row][0]
        label_len = int(self.data_cached[row][1])
        # value at idx=1 is the len of the label subtokens
        padding_start = int(data_len) + 2
        # label_padding_start = label_len + 1

        # Split data into raw data and padding part, concatenate in correct order.
        data_unpadded = self.data_cached[row][2:padding_start]
        padding = self.data_cached[row][padding_start:self.lookback_tokens]
        data = np.concatenate([data_unpadded, padding])

        # the label is the set of subtokens
        # We have the input seq till index = 999 => the label starts at 1000
        label_unpadded = self.data_cached[row][self.lookback_tokens:self.lookback_tokens+label_len]
        # Maybe set the end index to self.lookback_tokens + max_label_len
        label_padding = self.data_cached[row][self.lookback_tokens+label_len:]
        label = np.concatenate([label_unpadded, label_padding])
        # label = self.data_cached[row][self.lookback_tokens:]

        # return torch.LongTensor(data), data_len, label, label_len
        return torch.LongTensor(data), data_len, torch.LongTensor(label), label_len

    def __len__(self):
        """Returns the length of the dataset."""

        if self.mode.lower() == 'train':
            return 1970000
        elif self.mode.lower() == 'val':
            return 227255
        elif self.mode.lower() == 'test':
            return 911213
        else:
            raise RuntimeError("Unexpected dataset mode. "
                               "Supported modes are: train, val and test")

    def get_vocab_len(self):
        """Returns the length of the vocabulary."""
        return len(self.vocab)

    def get_mappings(self):
        return self.word2idx, self.idx2word






