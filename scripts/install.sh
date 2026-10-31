#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
python -c 'import sys; assert sys.version_info[:2] == (3, 9), "Use Python 3.9 (see environment.yml)"'
git submodule update --init --recursive
EXPECTED=4a388e64cd646ed7d7ad8de8fae55df2b8eea91d
ACTUAL=$(git -C third_party/fairseq rev-parse HEAD)
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
    echo "fairseq must be at $EXPECTED; found $ACTUAL" >&2
    exit 1
fi
python -m pip install 'pip==23.3.2' 'setuptools==59.5.0' 'wheel==0.38.4'
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation --editable ./third_party/fairseq
