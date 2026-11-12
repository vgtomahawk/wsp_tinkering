#!/bin/bash
# One-shot setup: submodules, venv, deps. Keys are read from the environment (ANTHROPIC_API_KEY, TINKER_API_KEY,
# optionally OPENROUTER_API_KEY); nothing secret is stored in the repo.
set -e
cd "$(dirname "$0")"
git submodule update --init --recursive
[ -d .venv ] || uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'mps', torch.backends.mps.is_available(), 'cuda', torch.cuda.is_available())"
echo "Gated model: accept the Llama 3.2 license on Hugging Face and run 'hf auth login' before the abliteration runs."
