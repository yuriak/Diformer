# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Diformer authors.
# Licensed under the MIT license in the root of this repository.
"""Directional Language Model and Diformer decoder."""

import torch
from torch import nn
from torch.nn import functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.cmlm_transformer import CMLMNATransformerModel, cmlm_base_architecture
from fairseq.models.nat.nonautoregressive_transformer import NATransformerDecoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from ..modules.directional_attention import DirectionalDecoderLayer

STRAIGHT, RIGHT, LEFT = 0, 1, 2


def directional_positions(tokens, pad):
    valid = tokens.ne(pad)
    right = valid.long().cumsum(1) * valid
    left = (valid.sum(1, keepdim=True) - right + 1) * valid
    return right, left


def causal_mask(tokens, direction):
    indices = torch.arange(tokens.size(1), device=tokens.device)
    blocked = indices[None, :] > indices[:, None]
    if direction == LEFT:
        blocked = blocked.transpose(0, 1)
    return blocked.unsqueeze(0).expand(tokens.size(0), -1, -1)


@register_model("diformer")
class DiformerModel(CMLMNATransformerModel):
    @staticmethod
    def add_args(parser):
        CMLMNATransformerModel.add_args(parser)
        parser.add_argument("--rel-pos-max-distance", type=int, default=256,
                            help="maximum clipped relative distance (default: 256)")

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if args.src_embedding_copy:
            raise ValueError("Diformer uses word-only K/V; source embedding copy is unsupported")
        if args.pred_length_offset:
            raise ValueError("Diformer predicts absolute target lengths; omit --pred-length-offset")
        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError("Diformer requires equal encoder and decoder embedding dimensions")
        decoder = DiformerDecoder(args, tgt_dict, embed_tokens)
        if args.apply_bert_init:
            decoder.apply(init_bert_params)
        return decoder

    def training_mask(self, tokens, directions):
        batch, length = tokens.shape
        content = tokens.ne(self.pad) & tokens.ne(self.bos) & tokens.ne(self.eos)
        # Independently choose a visible subset per query, always excluding itself.
        # Sample the visible context size uniformly for each query.
        scores = torch.rand(batch, length, length, device=tokens.device)
        diagonal = torch.eye(length, device=tokens.device, dtype=torch.bool)[None]
        scores.masked_fill_(diagonal, 2.0)
        scores.masked_fill_(~content[:, None, :], -1.0)
        rank = scores.argsort(-1).argsort(-1)
        boundary_count = (~content).sum(1)[:, None, None]
        visible_count = (
            content.sum(1)[:, None] * torch.rand(batch, length, device=tokens.device)
        ).long()[:, :, None]
        straight = rank >= boundary_count + visible_count
        straight = (straight | diagonal) & content[:, None, :]
        mask = torch.where(directions.eq(RIGHT)[:, :, None], causal_mask(tokens, RIGHT), straight)
        return torch.where(directions.eq(LEFT)[:, :, None], causal_mask(tokens, LEFT), mask)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)
        directions = torch.randint(3, tgt_tokens.shape, device=tgt_tokens.device)
        directions.masked_fill_(tgt_tokens.eq(self.bos), RIGHT)
        directions.masked_fill_(tgt_tokens.eq(self.eos), LEFT)
        directions.masked_fill_(tgt_tokens.eq(self.pad), STRAIGHT)
        right_targets = torch.cat((tgt_tokens[:, 1:], tgt_tokens.new_full((tgt_tokens.size(0), 1), self.pad)), 1)
        left_targets = torch.cat((tgt_tokens.new_full((tgt_tokens.size(0), 1), self.pad), tgt_tokens[:, :-1]), 1)
        targets = torch.where(directions.eq(RIGHT), right_targets, tgt_tokens)
        targets = torch.where(directions.eq(LEFT), left_targets, targets)
        targets = targets.masked_fill(tgt_tokens.eq(self.pad), self.pad)
        logits = self.decoder(
            normalize=False, encoder_out=encoder_out, prev_output_tokens=tgt_tokens,
            directions=directions, blocked=self.training_mask(tgt_tokens, directions),
        )
        return {
            "word_ins": {"out": logits, "tgt": targets, "mask": targets.ne(self.pad),
                         "ls": self.args.label_smoothing, "nll_loss": True},
            "length": {"out": length_out, "tgt": length_tgt,
                       "factor": self.decoder.length_loss_factor},
        }

    def directional_logits(self, tokens, encoder_out, direction):
        return self.decoder(
            normalize=False, encoder_out=encoder_out, prev_output_tokens=tokens,
            directions=torch.full_like(tokens, direction), blocked=causal_mask(tokens, direction),
        )


class DiformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn)
        dim = args.decoder_embed_dim
        self.direction_embedding = nn.Embedding(3, dim)
        self.right_embed_positions = nn.Embedding(self.max_target_positions + 1, dim)
        self.left_embed_positions = nn.Embedding(self.max_target_positions + 1, dim)
        self.relative_distance = args.rel_pos_max_distance
        if self.relative_distance < 1:
            raise ValueError("--rel-pos-max-distance must be positive")
        head_dim = dim // args.decoder_attention_heads
        self.k_rel_embed_positions = nn.Embedding(2 * self.relative_distance + 1, head_dim)
        self.v_rel_embed_positions = nn.Embedding(2 * self.relative_distance + 1, head_dim)
        self.embed_positions = None

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if no_encoder_attn:
            raise ValueError("Diformer requires encoder attention")
        if getattr(args, "checkpoint_activations", False):
            raise ValueError("Activation checkpointing is not supported by this extension")
        return DirectionalDecoderLayer(args, no_encoder_attn=False)

    def forward(self, normalize, encoder_out, prev_output_tokens, directions, blocked, **unused):
        tokens = prev_output_tokens
        right, left = directional_positions(tokens, self.padding_idx)
        positions = torch.where(
            directions.eq(LEFT).unsqueeze(-1),
            self.left_embed_positions(left), self.right_embed_positions(right),
        )
        x = (positions + self.direction_embedding(directions)).transpose(0, 1)
        words = self.embed_scale * self.embed_tokens(tokens)
        if self.project_in_dim is not None:
            words = self.project_in_dim(words)
        # No direction or absolute position may enter the fixed K/V stream.
        words = self.dropout_module(words).transpose(0, 1)
        indices = torch.arange(tokens.size(1), device=tokens.device)
        relative = (indices[None, :] - indices[:, None]).clamp(
            -self.relative_distance, self.relative_distance
        ) + self.relative_distance
        key_positions = self.k_rel_embed_positions(relative)
        value_positions = self.v_rel_embed_positions(relative)
        source_padding = encoder_out["encoder_padding_mask"]
        source_padding = source_padding[0] if source_padding else None
        for layer in self.layers:
            x = layer(x, words, key_positions, value_positions,
                      encoder_out["encoder_out"][0], source_padding,
                      blocked, tokens.eq(self.padding_idx))
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
        logits = self.output_layer(x)
        return F.log_softmax(logits, -1) if normalize else logits


@register_model_architecture("diformer", "diformer")
def diformer_base_architecture(args):
    cmlm_base_architecture(args)
