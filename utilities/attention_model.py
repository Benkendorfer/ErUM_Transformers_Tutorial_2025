"""
A simple attention model for the Enron email dataset.

This file contains the nuts and bolts of the attention model implementation
for the request classification task. It includes a simple attention
mechanism, a very basic encoder layer, and a model that stacks them together.

Written by Kees Benkendorfer and Knut Zoch for the 2025 ErUM-Data-Hub Deep
Learning tutorial in Aachen, Germany.
"""

import numpy as np

from torch import nn

import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Scaled dot-product attention function.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (batch_size, num_heads, seq_len, d_k).
    k : torch.Tensor
        Key tensor of shape (batch_size, num_heads, seq_len, d_k).
    v : torch.Tensor
        Value tensor of shape (batch_size, num_heads, seq_len, d_v).
    mask : torch.Tensor, optional
        Mask tensor of shape (batch_size, 1, seq_len, seq_len). Default is None.

    Returns
    -------
    torch.Tensor
        Output tensor of shape (batch_size, num_heads, seq_len, d_v).
    """
    scores = (q @ k.transpose(-2, -1)) / np.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn = F.softmax(scores, dim=0)

    # The @ symbol is used for matrix multiplication in PyTorch.
    return attn @ v


class SimpleAttention(nn.Module):
    """
    A simple attention mechanism.
    """

    def __init__(self, d_attention, d_embedding):
        super().__init__()
        self.q = nn.Linear(d_embedding, d_attention)
        self.k = nn.Linear(d_embedding, d_attention)
        self.v = nn.Linear(d_embedding, d_attention)
        self.out = nn.Linear(d_attention, d_embedding)

    def forward(self, x, mask=None, return_attn=False):
        """
        The forward pass of the attention mechanism.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_embedding).
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, seq_len). Default is None.
        return_attn : bool, optional
            If True, return the attention weights. Default is False.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_attention).
        torch.Tensor, optional
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
            Only returned if return_attn is True.
        """
        # x: [B, S, D], mask: Bool [B, S] True=real
        Q, K, V = self.q(x), self.k(x), self.v(x)
        scores  = (Q @ K.transpose(-2,-1)) / np.sqrt(Q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        attn = F.softmax(scores, dim=-1)             # [B, S, S]
        out  = self.out(attn @ V)                    # [B, S, D]
        return (out, attn) if return_attn else out


class SimpleEncoderLayer(nn.Module):
    """
    A simple encoder layer that consists of an attention mechanism and a
    feedforward network.
    """

    def __init__(self, d_attention, d_embedding, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = SimpleAttention(d_attention, d_embedding)
        self.norm1 = nn.LayerNorm(d_embedding)
        self.ff  = nn.Sequential(
            nn.Linear(d_embedding, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_embedding)
        )
        self.norm2 = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        """
        The forward pass of the encoder layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        mask : torch.Tensor, optional
            Mask tensor of shape (batch_size, seq_len). Default is None.
        return_attn : bool, optional
            If True, return the attention weights. Default is False.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
        torch.Tensor, optional
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
            Only returned if return_attn is True.
        """
        if return_attn:
            attn_out, attn = self.self_attn(x, mask=mask, return_attn=True)
        else:
            attn_out = self.self_attn(x, mask=mask)
            attn = None
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))

        if return_attn:
            return (x, attn)

        return x

class RequestClassifier(nn.Module):
    """
    A simple model that stacks multiple encoder layers to classify requests.
    """

    def __init__(self, vocab_size, d_attention=8, d_embedding=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embedding)
        self.layers = nn.ModuleList([
            SimpleEncoderLayer(d_attention, d_embedding) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_embedding, 2)

    def forward(self, src, src_key_padding_mask=None, return_attn=False):
        """
        Forward pass of the model.

        Parameters
        ----------
        src : torch.Tensor
            Input tensor of shape (batch_size, seq_len).
        src_key_padding_mask : torch.Tensor, optional
            Key padding mask tensor of shape (batch_size, seq_len). Default is None.
        return_attn : bool, optional
            If True, return the attention weights. Default is False.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 2).
        torch.Tensor, optional
            Attention weights of the last layer.
            They have shape (batch_size, num_heads, seq_len, seq_len).
            Only returned if return_attn is True.
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(src.size(0), src.size(1), dtype=torch.bool)
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
