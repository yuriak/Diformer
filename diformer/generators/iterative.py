# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Diformer authors.
# Licensed under the MIT license in the root of this repository.
"""Fixed-step refinement with optional bidirectional self-reranking."""

import torch

from ..models.diformer import LEFT, RIGHT, STRAIGHT
from .scoring import sequence_scores


class IterativeGenerator:
    def __init__(self, dictionary, strategy="mask_predict", steps=10, beam_size=5, self_rerank=False):
        if strategy not in ("mask_predict", "easy_first"):
            raise ValueError("Unknown NAR decoding strategy: " + strategy)
        if steps < 1 or beam_size < 1:
            raise ValueError("Refinement steps and length beam must be positive")
        self.dictionary = dictionary
        self.strategy = strategy
        self.steps = steps
        self.beam_size = beam_size
        self.self_rerank = self_rerank
        self.symbols_to_strip_from_output = {dictionary.bos(), dictionary.eos()}

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None, **kwargs):
        if len(models) != 1:
            raise ValueError("Diformer decoding expects exactly one model")
        if prefix_tokens is not None or constraints is not None:
            raise NotImplementedError("Constrained/prefix decoding is unsupported")
        model = models[0]
        model.eval()
        src = sample["net_input"]
        encoder_out = model.encoder(src["src_tokens"], src_lengths=src["src_lengths"])
        length_out = model.decoder.forward_length(normalize=True, encoder_out=encoder_out)
        lengths = model.decoder.forward_length_prediction(length_out, encoder_out)
        offsets = torch.arange(self.beam_size, device=lengths.device) - self.beam_size // 2
        # Absolute length classes are 0..255, including BOS/EOS.
        lengths = (lengths[:, None] + offsets).flatten().clamp(2, min(255, model.decoder.max_target_positions))
        order = torch.arange(src["src_tokens"].size(0), device=lengths.device).repeat_interleave(self.beam_size)
        encoder_out = model.encoder.reorder_encoder_out(encoder_out, order)
        positions = torch.arange(int(lengths.max()), device=lengths.device)[None]
        tokens = lengths.new_full((lengths.numel(), positions.size(1)), model.pad)
        tokens.masked_fill_(positions < lengths[:, None], model.unk)
        tokens[:, 0] = model.bos
        tokens.scatter_(1, lengths[:, None] - 1, model.eos)
        content = positions.gt(0) & positions.lt(lengths[:, None] - 1)
        padding = tokens.eq(model.pad)
        scores = torch.zeros_like(tokens, dtype=torch.float)
        easy_mask = None

        for step in range(self.steps):
            if self.strategy == "mask_predict":
                update = tokens.eq(model.unk) & content
                blocked = update[:, None, :].expand(-1, tokens.size(1), -1)
            else:
                update = content
                if step == 0:
                    blocked = content[:, None, :].expand(-1, tokens.size(1), -1)
                else:
                    if easy_mask is None:
                        rank = scores.masked_fill(~content, float("-inf")).argsort(-1, descending=True).argsort(-1)
                        # Only tokens strictly easier in iteration 1 are visible.
                        easy_mask = (rank[:, None, :] >= rank[:, :, None]) & content[:, None, :]
                    blocked = easy_mask
            log_probs = model.decoder(
                normalize=True, encoder_out=encoder_out, prev_output_tokens=tokens,
                directions=torch.full_like(tokens, STRAIGHT), blocked=blocked,
            ).float()
            # Structural tokens cannot replace words or create premature endpoints.
            log_probs[:, :, [model.pad, model.bos, model.eos, model.unk]] = float("-inf")
            new_scores, predictions = log_probs.max(-1)
            tokens = torch.where(update, predictions, tokens)
            scores = torch.where(update, new_scores, scores)
            if self.strategy == "mask_predict" and step + 1 < self.steps:
                count = (content.sum(1) * (1 - (step + 1) / self.steps)).long()
                sorted_indices = scores.masked_fill(~content, float("inf")).argsort(-1)
                low_confidence = positions < count[:, None]
                remask = torch.zeros_like(content).scatter(1, sorted_indices, low_confidence)
                tokens.masked_fill_(remask, model.unk)

        if self.self_rerank:
            candidate_scores = (
                sequence_scores(model, tokens, encoder_out, RIGHT)
                + sequence_scores(model, tokens, encoder_out, LEFT)
            ) / 2
        else:
            candidate_scores = scores.masked_fill(~content, 0).sum(1) / content.sum(1).clamp_min(1)
        finalized = []
        for sentence in range(src["src_tokens"].size(0)):
            candidates = []
            for beam in range(self.beam_size):
                row = sentence * self.beam_size + beam
                valid = ~padding[row]
                valid[0] = False  # fairseq hypotheses omit BOS and retain EOS.
                candidates.append({
                    "tokens": tokens[row, valid], "score": candidate_scores[row],
                    "positional_scores": scores[row, valid], "attention": None,
                    "alignment": None, "steps": self.steps,
                })
            finalized.append(sorted(candidates, key=lambda h: float(h["score"]), reverse=True))
        return finalized
