"""
Data-loading utilities for the Enron email dataset.

This file contains the EnronRequestDataset class, which is a PyTorch Dataset
for the Enron email dataset. It is used to load the data and prepare it for
training an attention model.

Written by Kees Benkendorfer and Knut Zoch for the 2025 ErUM-Data-Hub Deep
Learning tutorial in Aachen, Germany.
"""

import torch
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, vocab: Vocab):
    """
    Collation function for the EnronRequestDataset.

    This function takes a batch of data and pads the sequences to the
    maximum length in the batch. It also creates a mask for the padding
    tokens.
    """

    token_ids_list, labels = zip(*batch)
    # pad to batch max‐len
    src = pad_sequence(token_ids_list,
                       batch_first=True,
                       padding_value=vocab['<pad>'])
    labels = torch.tensor(labels, dtype=torch.long)
    # pad_mask: True for PAD tokens, False for real tokens
    pad_mask = src == vocab['<pad>']
    return src, labels, pad_mask

class EnronRequestDataset(Dataset):
    """
    Class to load the Enron email dataset for training an attention model.

    This class inherits from PyTorch's Dataset class and implements the
    __len__ and __getitem__ methods to provide the data in a format suitable
    for training. The dataset consists of email texts and their corresponding
    labels (0 for no request, 1 for request). The texts are tokenized and
    converted to numerical IDs using a user-provided vocabulary.
    """

    def __init__(
            self,
            texts: list[str],
            labels: list[int],
            vocab: Vocab,
            tokenizer
        ):
        """
        Constructor for the EnronRequestDataset.

        Parameters
        ----------
        texts : list[str]
            List of email texts.
        labels: list[int]
            List of labels for email texts. (0=no-request, 1=request)
        vocab: torchtext.vocab.Vocab
            Vocabulary object to convert tokens to numerical IDs.
        tokenizer:
            Tokenizer for text.
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        token_ids = torch.tensor(self.vocab(tokens), dtype=torch.long)
        label    = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_ids, label
