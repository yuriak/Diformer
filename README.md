# Diformer

**One translation model, three generation directions, four decoding strategies.**

Source code for **Diformer: Directional Transformer for Neural Machine Translation**
(EAMT 2022), by Minghan Wang, Jiaxin Guo, Yuxia Wang, Daimeng Wei, Hengchao Shang,
Yinglu Li, Chang Su, Yimeng Chen, Min Zhang, Shimin Tao, and Hao Yang.

[Paper](https://aclanthology.org/2022.eamt-1.11/) ·
[PDF](https://aclanthology.org/2022.eamt-1.11.pdf) ·
[arXiv](https://arxiv.org/abs/2112.11632)

Diformer jointly learns left-to-right (L2R), right-to-left (R2L), and straight
generation with a Directional Language Model objective. At inference time, the
same checkpoint supports autoregressive beam search in either direction and
non-autoregressive translation with mask-predict or parallel easy-first.
Its two autoregressive directions can also score its own candidates for
self-reranking, without a separate reranker model.

## Method

During training, each target position independently samples one of three
directions, with BOS fixed to L2R and EOS fixed to R2L. The input sentence stays
in its original order; the prediction target shifts according to the sampled
direction:

| Direction | Prediction from input position `i` | Visible target context |
| --- | --- | --- |
| L2R | Token at `i + 1` | Positions up to and including `i` |
| R2L | Token at `i - 1` | Positions from `i` onward |
| Straight | Token at `i` | A sampled subset excluding `i` |

The decoder combines direction embeddings and separately parameterized forward
and backward position embeddings to initialize its queries. Every decoder layer
uses the same decontextualized word embeddings for keys and values. Clipped
relative position embeddings restore positional information inside attention,
without leaking absolute direction-dependent positions through the K/V stream.

Training minimizes label-smoothed token cross-entropy plus `0.1` times the
target-length cross-entropy. Length prediction is used only by NAR decoding;
AR generation terminates when the appropriate boundary token is predicted.

## Installation

Use Linux, Python **3.9**, PyTorch **1.13.1**, and fairseq **v0.12.2**.

```bash
git clone --recurse-submodules https://github.com/yuriak/Diformer.git
cd Diformer
conda env create -f environment.yml
conda activate diformer
bash scripts/install.sh
```

The installer pins pip/setuptools/wheel before installing the Python
requirements and the editable fairseq submodule. Older OmegaConf metadata
requires the pinned pip version. For GPU training, use a PyTorch 1.13.1 build
appropriate for your CUDA installation. A compiler is needed to build fairseq's
extensions.

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

The submodule pins `4a388e64cd646ed7d7ad8de8fae55df2b8eea91d`. See
[third_party/README.md](third_party/README.md) for submodule setup.

## Data preparation

Diformer trains on WMT14 English↔German and WMT16 English↔Romanian with
Moses tokenization, BPE, and sequence-level knowledge-distilled (KD) data.
The teacher is Transformer-big for En↔De and Transformer-base for
En↔Ro. For each translation direction, prepare teacher translations of the
training sources as the training targets. Use the original human references
for validation and test evaluation.

Prepare Moses-tokenized, BPE-encoded files with the following layout:

```text
data/wmt14.en-de/
├── train.en
├── train.de       # teacher translations for KD training
├── valid.en
├── valid.de       # human references
├── test.en
└── test.de        # human references
```

Use [Moses](https://github.com/moses-smt/mosesdecoder) for tokenization and
[subword-nmt](https://github.com/rsennrich/subword-nmt) for joint BPE. Learn BPE
on training data only and apply the same codes to validation/test data. Keep
the original tokenized, **non-BPE** test references separately for scoring.
For Romanian, use the dataset's documented normalization and diacritic removal
conventions consistently for teacher data, training, and evaluation.

Binarize the prepared BPE text:

```bash
bash scripts/preprocess.sh en de data/wmt14.en-de data-bin/wmt14.en-de
```

This builds a joined dictionary, as required by shared source/target embeddings.
For De→En, reverse the language arguments and supply the corresponding KD
targets; the distilled targets from En→De do not serve as De→En training data.
For En↔Ro, use `en ro` or `ro en` with the same file conventions.

Do not add BOS/EOS as literal text. The fairseq translation task adds sentence
boundaries and right-pads targets for directional training.

## Training

```bash
# Effective batch: 8 GPUs × 8192 tokens × 2 accumulation steps.
GPUS=8 bash scripts/train.sh wmt14_en_de \
  data-bin/wmt14.en-de en de checkpoints/en-de --fp16

GPUS=8 bash scripts/train.sh wmt16_en_ro \
  data-bin/wmt16.en-ro en ro checkpoints/en-ro --fp16
```

The same presets support the reverse language directions. Adjust `GPUS`,
`MAX_TOKENS`, and `UPDATE_FREQ` for your hardware. For example, the same nominal
token budget on one GPU uses `GPUS=1 MAX_TOKENS=8192 UPDATE_FREQ=16`.
Actual non-padding tokens vary with batch packing. `--fp16` is opt-in.

| Setting | Value |
| --- | --- |
| Encoder / decoder layers | 6 / 6 |
| Hidden / FFN dimension | 512 / 2048 |
| Attention heads | 8 |
| Direction sampling | Uniform over three directions |
| Straight context | Disentangled, excludes the query token |
| Maximum relative distance | 256 |
| Length-loss weight | 0.1 |
| Optimizer | Adam, peak LR `5e-4` |
| LR schedule | 10k warmup, inverse square root decay |
| Effective token budget | 131,072 (~128k) |
| En↔De / En↔Ro training updates | 300k / 100k |
| Dropout / label smoothing | 0.3 / 0.1 |
| Weight decay / Adam betas | 0.01 / `(0.9, 0.98)` |
| Checkpoint interval | Every 1,000 updates |
| Final checkpoint | Average of last 5 update checkpoints |

Additional `fairseq-train` arguments can be appended to the command. For a
shorter training recipe, override the update count with `TRAIN_UPDATES`.
The scripts choose the standard Transformer-size architecture explicitly and
use fairseq's built-in `nat_loss` criterion.

Average the final five update checkpoints before evaluation:

```bash
bash scripts/average_checkpoints.sh checkpoints/en-de checkpoints/en-de/average.pt
```

The checkpoint save interval must leave at least five update checkpoints to
average. Epoch checkpoints are disabled; the averaging script selects update
checkpoints rather than copies such as `checkpoint_last.pt`.

## Translation

All four strategies use the same trained checkpoint:

```bash
bash scripts/generate.sh data-bin/wmt14.en-de checkpoints/en-de/average.pt \
  en de l2r results/en-de/l2r

bash scripts/generate.sh data-bin/wmt14.en-de checkpoints/en-de/average.pt \
  en de r2l results/en-de/r2l

bash scripts/generate.sh data-bin/wmt14.en-de checkpoints/en-de/average.pt \
  en de mask_predict results/en-de/mask-predict

bash scripts/generate.sh data-bin/wmt14.en-de checkpoints/en-de/average.pt \
  en de easy_first results/en-de/easy-first
```

AR beam size defaults to 4. NAR length beam size defaults to 5 and refinement
uses exactly 10 decoder passes, with no early stopping. Easy-first determines
its confidence ordering from the first prediction and keeps it fixed.
R2L output is restored to normal reading order before writing hypotheses.
The absolute target-length classifier has 256 classes; NAR candidates are
clamped to 2–255 tokens including BOS/EOS.

Useful environment overrides:

| Variable | Default | Meaning |
| --- | --- | --- |
| `AR_BEAM` | 4 | AR beam width |
| `LENGTH_BEAM` | 5 | NAR neighboring-length candidates |
| `STEPS` | 10 | NAR passes without reranking |
| `BATCH_SIZE` | 32 | Generation batch size |
| `SUBSET` | `test` | Dataset split |
| `NBEST` | 1 | Number of hypotheses to write |
| `SELF_RERANK` | 0 | Enable scoring with the same model |
| `REFINEMENT_STEPS` | 8 | NAR passes when self-reranking is enabled |

For self-reranking:

```bash
SELF_RERANK=1 bash scripts/generate.sh \
  data-bin/wmt14.en-de checkpoints/en-de/average.pt \
  en de mask_predict results/en-de/mask-predict-rerank
```

NAR self-reranking performs **8 refinement passes + 2 scoring passes**, reusing
the encoder states and averaging length-normalized L2R and R2L scores.
The option also works with easy-first. For AR, `SELF_RERANK=1` reranks
the completed beam using the opposite direction's length-normalized score.
AR beam search evaluates the full prefix at each generation step.

Generation options are task settings, independent of the checkpoint's training
configuration. Direct CLI use is also possible:

```bash
fairseq-generate data-bin/wmt14.en-de --user-dir ./diformer \
  --task diformer_translation --path checkpoints/en-de/average.pt \
  --source-lang en --target-lang de --decode-strategy easy_first \
  --length-beam 5 --refinement-steps 8 --self-rerank --remove-bpe
```

Set `--refinement-steps N` for exactly N NAR decoder calls and `--self-rerank`
to score candidates with the same model.

## BLEU evaluation

`generate.sh` removes `@@` BPE markers but preserves Moses tokenization. Score
against the corresponding tokenized, non-BPE reference file:

```bash
bash scripts/evaluate.sh results/en-de/l2r/generate-test.txt \
  data/references/test.de.tok
```

The extraction script restores sentence order and keeps the first hypothesis
per sentence, including when `NBEST` is greater than 1. Evaluation uses
SacreBLEU with `--tokenize none` for already tokenized text. Keep its signature
with reported results.

## Code layout

```text
diformer/
  models/diformer.py               Direction sampling, target shifts, decoder
  modules/directional_attention.py Relative K/V attention and decoder layer
  tasks/translation.py             fairseq registration and decoding settings
  generators/iterative.py           Mask-predict, easy-first, self-reranking
  generators/autoregressive.py      L2R/R2L adapters to fairseq beam search
  generators/scoring.py             Directional sequence likelihoods
configs/                           WMT training presets
scripts/                           Installation, data, training and evaluation
third_party/fairseq/                Pinned, unmodified dependency submodule
```

The public entry points are `--arch diformer`, `--task diformer_translation`,
and `--user-dir ./diformer`.

## Citation

If you use this code or method, please cite the EAMT paper:

```bibtex
@inproceedings{wang-etal-2022-diformer,
    title = "Diformer: Directional Transformer for Neural Machine Translation",
    author = "Wang, Minghan and Guo, Jiaxin and Wang, Yuxia and Wei, Daimeng and
              Shang, Hengchao and Li, Yinglu and Su, Chang and Chen, Yimeng and
              Zhang, Min and Tao, Shimin and Yang, Hao",
    booktitle = "Proceedings of the 23rd Annual Conference of the European Association for Machine Translation",
    month = jun,
    year = "2022",
    address = "Ghent, Belgium",
    publisher = "European Association for Machine Translation",
    url = "https://aclanthology.org/2022.eamt-1.11/",
    pages = "81--90"
}
```

## License

Code is released under the [MIT License](LICENSE). fairseq-derived code retains
the upstream copyright notices; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The linked paper and external datasets retain their own licenses.
