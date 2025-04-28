import math

import torch.nn as nn

import torch.nn.functional as F

import torch
from torch.utils.data import Dataset

def scaled_dot_product_attention(q, k, v, mask=None):
    # q,k,v: [batch, heads, seq, d_k]
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return attn @ v


class SimpleAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, return_attn=False):
        # x: [B, S, D], mask: Bool [B, S] True=real
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores  = (Q @ K.transpose(-2,-1)) / math.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)             # [B, S, S]
        out  = self.out(attn @ V)                    # [B, S, D]
        return (out, attn) if return_attn else out

# Now a tiny encoder layer: attention + FFN + residual & norm
class MyEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = SimpleAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff  = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        if return_attn:
            attn_out, attn = self.self_attn(x, mask=mask, return_attn=True)
        else:
            attn_out = self.self_attn(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return (x, attn) if return_attn else x

# Stack N layers and attach classifier
class RequestClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MyEncoderLayer(d_model) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, src, src_key_padding_mask=None, return_attn=False):
        pad_mask   = src_key_padding_mask
        valid_mask = ~pad_mask
        x = self.embed(src)
        last_attn = None
        for layer in self.layers:
            if return_attn:
                x, last_attn = layer(x, mask=valid_mask, return_attn=True)
            else:
                x = layer(x, mask=valid_mask)
        lengths = valid_mask.sum(1, keepdim=True).clamp(min=1)
        pooled  = (x * valid_mask.unsqueeze(-1)).sum(1) / lengths
        logits = self.classifier(pooled)
        return (logits, last_attn) if return_attn else logits

class EnronRequestDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer):
        """
        texts: List[str]
        labels: List[int]  (0=no-request, 1=request)
        vocab: torchtext.vocab.Vocab
        tokenizer: callable→List[str]
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

