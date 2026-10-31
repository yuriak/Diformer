# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Diformer authors.
# Licensed under the MIT license in the root of this repository.
"""Shaw relative positions with fixed, decontextualized word keys/values."""

import torch
from fairseq.modules import MultiheadAttention
from fairseq.modules.transformer_layer import TransformerDecoderLayer


class DirectionalAttention(MultiheadAttention):
    def forward(self, query, key, value, key_positions, value_positions,
                blocked, key_padding_mask):
        # Q: T x B x C; relative embeddings: T x T x head_dim.
        length, batch, channels = query.shape

        def split_heads(x):
            return x.reshape(length, batch, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        q = split_heads(self.q_proj(query)) * self.scaling
        k = split_heads(self.k_proj(key))
        v = split_heads(self.v_proj(value))
        logits = q @ k.transpose(-1, -2)
        logits = logits + torch.einsum("bhid,ijd->bhij", q, key_positions)
        # Keep masks boolean and normalize in float32, including under fp16.
        mask = blocked[:, None] | key_padding_mask[:, None, None, :]
        weights = logits.float().masked_fill(mask, float("-inf")).softmax(-1).to(q.dtype)
        weights = self.dropout_module(weights)
        result = weights @ v + torch.einsum("bhij,ijd->bhid", weights, value_positions)
        result = result.permute(2, 0, 1, 3).reshape(length, batch, channels)
        return self.out_proj(result)


class DirectionalDecoderLayer(TransformerDecoderLayer):
    """Use evolving queries but the same input word embeddings in every layer."""

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return DirectionalAttention(
            embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True,
            q_noise=self.quant_noise, qn_block_size=self.quant_noise_block_size,
        )

    def forward(self, x, words, key_positions, value_positions,
                encoder_out, encoder_padding_mask, blocked, padding_mask):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, words, words, key_positions, value_positions, blocked, padding_mask)
        x = self.residual_connection(self.dropout_module(x), residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_padding_mask, need_weights=False,
        )
        x = self.residual_connection(self.dropout_module(x), residual)
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_dropout_module(self.activation_fn(self.fc1(x)))
        x = self.fc2(x)
        x = self.residual_connection(self.dropout_module(x), residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x
