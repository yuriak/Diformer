# Third-party notices

## fairseq

This project is built on [fairseq](https://github.com/facebookresearch/fairseq),
Copyright (c) Facebook, Inc. and its affiliates, distributed under the MIT
license. The unmodified dependency is included as a Git submodule at
`third_party/fairseq`, pinned to v0.12.2, commit
`4a388e64cd646ed7d7ad8de8fae55df2b8eea91d`.

The model, directional decoder layer, and generation code build on
fairseq's Transformer, conditional masked
language model, iterative refinement, and sequence generation implementations.
Their copyright notices are retained in the corresponding files, and the MIT
license text is included in `LICENSE`. The upstream license is also available
in `third_party/fairseq/LICENSE` after initializing the submodule.

Other dependencies retain their respective licenses. Moses and subword-nmt
are preprocessing tools distributed under their respective licenses. The
paper and datasets retain their own licenses.
