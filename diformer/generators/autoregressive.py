# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Diformer authors.
# Licensed under the MIT license in the root of this repository.
"""Use fairseq beam search for both directions, restoring canonical output order."""

import torch
from fairseq.data import data_utils
from fairseq.sequence_generator import EnsembleModel, SequenceGenerator

from ..models.diformer import LEFT, RIGHT
from .scoring import sequence_scores


class DirectionalEnsemble(EnsembleModel):
    def __init__(self, models, direction):
        if len(models) != 1:
            raise ValueError("Diformer decoding expects exactly one model")
        super().__init__(models)
        self.direction = direction

    def forward_decoder(self, tokens, encoder_outs, incremental_states, temperature=1.0):
        model = self.models[0]
        context = tokens.clone()
        if self.direction == RIGHT:
            context[:, 0] = model.bos
        else:
            context = context.flip(1)  # [EOS, last, ...] -> [..., last, EOS].
        logits = model.directional_logits(context, encoder_outs[0], self.direction)
        logits = logits[:, -1 if self.direction == RIGHT else 0].float() / temperature
        if self.direction == LEFT:
            # Beam search uses EOS as its terminator; the model predicts BOS.
            logits[:, [model.bos, model.eos]] = logits[:, [model.eos, model.bos]]
        log_probs = logits.log_softmax(-1)
        log_probs[:, model.bos] = float("-inf")
        return log_probs, None

    def reorder_incremental_state(self, incremental_states, new_order):
        # Directional positions are recomputed on the full prefix, without KV caches.
        return None


class AutoregressiveGenerator(SequenceGenerator):
    def __init__(self, models, dictionary, direction, self_rerank=False, **kwargs):
        super().__init__(DirectionalEnsemble(models, direction), dictionary, **kwargs)
        self.direction = direction
        self.self_rerank = self_rerank
        self.symbols_to_strip_from_output = {dictionary.bos(), dictionary.eos()}

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None, **kwargs):
        if prefix_tokens is not None or constraints is not None:
            raise NotImplementedError("Constrained/prefix decoding is unsupported")
        hypotheses = super().generate(models, sample, **kwargs)
        if self.direction == LEFT:
            for sentence in hypotheses:
                for hypothesis in sentence:
                    # Last token is logical EOS (the model's generated BOS).
                    tokens = hypothesis["tokens"]
                    hypothesis["tokens"] = torch.cat((tokens[:-1].flip(0), tokens[-1:]))
                    scores = hypothesis["positional_scores"]
                    hypothesis["positional_scores"] = torch.cat((scores[:-1].flip(0), scores[-1:]))
        if self.self_rerank:
            self.rerank(models[0], hypotheses, sample)
        return hypotheses

    def rerank(self, model, hypotheses, sample):
        # Re-encode the original batch once, then explicitly map every candidate
        # to its source. This also handles batches whose sentences finish early.
        src = sample["net_input"]
        encoder_out = model.encoder(src["src_tokens"], src_lengths=src["src_lengths"])
        rows, order = [], []
        for sentence, candidates in enumerate(hypotheses):
            for hypothesis in candidates:
                tokens = hypothesis["tokens"]
                rows.append(torch.cat((tokens.new_tensor([model.bos]), tokens)))
                order.append(sentence)
        if not rows:
            return
        tokens = data_utils.collate_tokens(rows, model.pad, left_pad=False)
        order = tokens.new_tensor(order)
        encoder_out = model.encoder.reorder_encoder_out(encoder_out, order)
        reverse_direction = LEFT if self.direction == RIGHT else RIGHT
        scores = sequence_scores(model, tokens, encoder_out, reverse_direction)
        row = 0
        for candidates in hypotheses:
            for hypothesis in candidates:
                hypothesis["score"] = scores[row]
                row += 1
            candidates.sort(key=lambda h: float(h["score"]), reverse=True)
