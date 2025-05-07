"""
Visualization functions for the request classification task.

This file contains several visualization utilities for the attention model
used in the request classification task.

Written by Kees Benkendorfer and Knut Zoch for the 2025 ErUM-Data-Hub Deep
Learning tutorial in Aachen, Germany.
"""

from IPython.display import HTML, display
import numpy as np

import matplotlib.pyplot as plt


def display_sentence_with_alpha(sentence, tokenizer, attention):
    """
    Render a sentence with varying opacity based on attention scores.
    """
    tokens = tokenizer(sentence)

    # Scores are sum of query weights for each token
    scores = np.sum(attention, axis=0)

    # Normalize scores to [0, 1]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        scores = [1.0] * len(scores)  # All scores are the same
    else:
        scores = [(score - min_score) / (max_score - min_score) for score in scores]

    display_tokens_with_alpha(tokens, scores)


def display_tokens_with_alpha(tokens, scores):
    """
    Render tokens in a Jupyter notebook with varying opacity.

    tokens: List[str]
    scores: List[float] (0.0 transparent → 1.0 opaque)
    """
    # Normalize scores to [0, 1]
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        scores = [1.0] * len(scores)  # All scores are the same
    else:
        scores = [(score - min_score) / (max_score - min_score) for score in scores]

    # Display with full opacity right above the partial opacity on the same line
    spans = "".join(
        f'<span style="opacity: 1.0; margin-right: 4px;">{token}</span>'
        for token in tokens
    )
    display(HTML(spans))

    # And now with varying opacity
    spans = "".join(
        f'<span style="opacity: {score:.2f}; margin-right: 4px;">{token}</span>'
        for token, score in zip(tokens, scores)
    )
    display(HTML(spans))


def plot_attention(tokens, attn: np.ndarray):
    """
    Visualize the attention weights in a simple plot.

    Parameters
    ----------
    tokens : List[str]
        The tokens to display on the axes.
    attn : np.ndarray
        The attention weights, a 2D NumPy array of shape (S, S) where S is the
        number of tokens.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(attn.T)
    fig.colorbar(cax)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    plt.ylabel("Key positions (attended by)")
    plt.xlabel("Query positions (attending to)")
    plt.show()
