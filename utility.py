from IPython.display import HTML, display
import numpy as np

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

