"""Score completed, canonical [BOS, words..., EOS] sequences."""

from ..models.diformer import LEFT, RIGHT


def sequence_scores(model, tokens, encoder_out, direction):
    log_probs = model.directional_logits(tokens, encoder_out, direction).float().log_softmax(-1)
    if direction == RIGHT:
        targets = tokens[:, 1:]
        selected = log_probs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        valid = targets.ne(model.pad)
    elif direction == LEFT:
        targets = tokens[:, :-1]
        selected = log_probs[:, 1:].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        valid = targets.ne(model.pad) & targets.ne(model.eos) & tokens[:, 1:].ne(model.pad)
    else:
        raise ValueError("Sequence scoring requires an autoregressive direction")
    return selected.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)
