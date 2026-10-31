# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Diformer authors.
# Licensed under the MIT license in the root of this repository.
"""Translation task with explicit generation settings independent of checkpoints."""

from dataclasses import dataclass, field

import torch

from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinConfig, TranslationLevenshteinTask

from ..generators.autoregressive import AutoregressiveGenerator
from ..generators.iterative import IterativeGenerator
from ..models.diformer import LEFT, RIGHT

DECODE_CHOICES = ChoiceEnum(["l2r", "r2l", "mask_predict", "easy_first"])


@dataclass
class DiformerTranslationConfig(TranslationLevenshteinConfig):
    decode_strategy: DECODE_CHOICES = field(default="mask_predict", metadata={"help": "decoding strategy"})
    refinement_steps: int = field(default=10, metadata={"help": "exact number of NAR decoder passes"})
    length_beam: int = field(default=5, metadata={"help": "number of neighboring target lengths"})
    self_rerank: bool = field(default=False, metadata={"help": "score candidates using this model's AR directions"})


@register_task("diformer_translation", dataclass=DiformerTranslationConfig)
class DiformerTranslationTask(TranslationLevenshteinTask):
    def valid_step(self, sample, model, criterion):
        loss, sample_size, logs = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            # translation_lev's valid_step omits TranslationTask's BLEU path.
            with torch.no_grad():
                bleu_sample = dict(sample)
                targets = sample["target"].clone()
                targets.masked_fill_(targets.eq(self.tgt_dict.bos()), self.tgt_dict.pad())
                bleu_sample["target"] = targets
                bleu = self._inference_with_bleu(self.sequence_generator, bleu_sample, model)
            logs["_bleu_sys_len"] = bleu.sys_len
            logs["_bleu_ref_len"] = bleu.ref_len
            for i in range(4):
                logs["_bleu_counts_" + str(i)] = bleu.counts[i]
                logs["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logs

    def build_generator(self, models, args, **kwargs):
        if getattr(args, "iter_decode_with_external_reranker", False):
            raise ValueError("Use --self-rerank; Diformer needs no external reranker")
        strategy = self.cfg.decode_strategy
        if strategy in ("l2r", "r2l"):
            return AutoregressiveGenerator(
                models, self.target_dictionary, RIGHT if strategy == "l2r" else LEFT,
                self_rerank=self.cfg.self_rerank,
                beam_size=getattr(args, "beam", 4),
                max_len_a=getattr(args, "max_len_a", 0.0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                len_penalty=getattr(args, "lenpen", 1.0),
                temperature=getattr(args, "temperature", 1.0),
            )
        return IterativeGenerator(
            self.target_dictionary, strategy=strategy, steps=self.cfg.refinement_steps,
            beam_size=self.cfg.length_beam, self_rerank=self.cfg.self_rerank,
        )
