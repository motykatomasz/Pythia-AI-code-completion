import os
import torch
import torch.utils.data as data
import numpy as np

from utils.utils import get_files, read_csv

UNKNOWN_TOKEN = '<UNK>'
PADDING_TOKEN = '<PAD>'


def add_unknown_token(vocab):
    return np.append(vocab, UNKNOWN_TOKEN)


def add_padding_token(vocab):
    return np.append(vocab, PADDING_TOKEN)


def create_mappings(vocab):
    word2idx = {w: idx for (idx, w) in enumerate(vocab)}
    idx2word = {idx: w for (idx, w) in enumerate(vocab)}
    type2idx, idx2type = create_type_mapping(vocab)

    return word2idx, idx2word, type2idx, idx2type


def create_type_mapping(vocab):
    type2idx = {}
    idx2type = {}
    for (idx, w) in enumerate(vocab):
        t = w.split(':')
        if len(t) > 1:
            type = t[1]
            idx2type[idx] = type
            if type in type2idx:
                type2idx[type].append(idx)
            else:
                type2idx[type] = [idx]
        else:
            idx2type[idx] = None

    return type2idx, idx2type


class Python150k(data.Dataset):
    """Python150k dataset loader.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    - context_size (``int``): Number of tokens before and after the target token
    """
    # TODO Store train/validation/test data in different files and load them separately
    ast_train_dir = 'training'
    ast_val_dir = 'validation'
    ast_test_dir = 'test'
    vocab_file = 'voc.npy'

    def __init__(self,
                 root_dir,
                 mode='train',
                 lookback_tokens=100,
                 chunk_size=10000, max_len_label=20):
        self.root_dir = root_dir
        self.mode = mode
        self.idx_cached = -1
        self.chunk_size = chunk_size
        self.lookback_tokens = lookback_tokens
        self.max_len_label = max_len_label

        self.vocab = self.init_vocab(self.root_dir)

        self.word2idx, self.idx2word, self.type2idx, self.idx2type = create_mappings(self.vocab)
        self.padding_idx = self.word2idx[PADDING_TOKEN]

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

    def init_vocab(self, root_dir):
        vocab = np.load(os.path.join(root_dir, self.vocab_file))
        vocab = add_unknown_token(vocab)
        vocab = add_padding_token(vocab)

        return vocab

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
        padding_start = int(data_len) + 1

        # Split data into raw data and padding part, concatenate in correct order.
        data_unpadded = self.data_cached[row][1:padding_start]
        padding = self.data_cached[row][padding_start:self.lookback_tokens+1]
        data = np.concatenate([data_unpadded, padding])

        label_len = self.data_cached[row][1]
        label_padding_start = int(label_len) + 1

        label_unpadded = self.data_cached[row][self.lookback_tokens + 1:label_padding_start]
        # probably add a max_len_label param in the trainset
        label_padding = self.data_cached[row][label_padding_start:self.max_len_label + 1]
        label_data = np.concatenate([label_unpadded, label_padding])

        return torch.LongTensor(data), data_len, torch.LongTensor(label_data)

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
        return self.word2idx, self.idx2word, self.type2idx, self.idx2type






