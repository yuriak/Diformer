# Implementation notes

Diformer combines a joint directional objective, directional positions,
word-only K/V attention, relative position embeddings, four decoding strategies,
and self-reranking.

## Training and information flow

- Targets contain BOS and EOS and are right-padded. Directions are independently
  sampled with probability 1/3; BOS is fixed to L2R and EOS to R2L.
- An L2R query at `i` predicts `i+1` and can see `j <= i`. An R2L query predicts
  `i-1` and can see `j >= i`. The current input token is valid AR context.
- A straight query predicts its own input position, so its own word must always
  be masked. The remaining context size and visible subset are randomly sampled
  independently per query. BOS/EOS remain visible; padding keys never are.
- Queries start from position plus direction embeddings. The same word-only
  representations are used as K/V in every layer. Hidden query states never
  enter the K/V stream, so another query cannot relay a masked token across
  layers. Relative position embeddings are added within attention only.
- Attention masks stay boolean and are applied before float32 softmax.

## Generation

- NAR's `refinement_steps` is an exact number of passes, including the initial
  all-mask prediction. Remasking happens only when another pass remains.
- Mask-predict selects low-confidence content positions only. BOS, EOS, and
  padding cannot be selected for remasking or replace predicted content.
- Easy-first establishes its rank from the initial prediction. Each content
  query sees strictly higher-ranked content tokens and boundary tokens. It does
  not see itself, and the ordering remains fixed during subsequent passes.
- Scores are log probabilities in both NAR strategies. Confidence ranking is
  identical to ranking probabilities, while sequence comparisons stay in one
  consistent score space. Without reranking, candidates use the mean content
  log probability, excluding structural boundaries and padding.
- Length candidates and encoder states share an explicit repeated sentence
  index. All candidates undergo the same number of refinement passes.
- NAR self-reranking reuses these states, computes the mean non-padding token
  log probability in each AR direction (including the predicted endpoint),
  and averages the two scores. Defaults are eight refinement passes and two
  scoring passes. No extra checkpoint is loaded.
- AR beam search uses fairseq's generator with a directional adapter. R2L
  prefixes are reversed into canonical order before model evaluation. BOS/EOS
  output labels are exchanged at the search interface so ordinary EOS-based
  beam termination represents the model's R2L prediction of BOS. Completed
  R2L hypotheses and their positional scores return to normal reading order.
- AR self-reranking scores the completed beam in the opposite direction. A
  fresh encoder batch is explicitly expanded to match each source's candidates;
  its score replaces the search score for ordering candidates.

## Dependencies

Diformer uses fairseq v0.12.2, commit
`4a388e64cd646ed7d7ad8de8fae55df2b8eea91d`, for the Transformer encoder,
NAT length prediction, translation data loading, training criterion, and
autoregressive beam search. The directional decoder and generation adapters
are implemented in the `diformer` package.
