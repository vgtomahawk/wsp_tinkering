"""Run weight-space-bench test 1 (Exp 5: write MinAtar networks from scratch) with our model list.

  python -m wspt.run_minatar run claude-opus-5.5-direct     # ask + score; resumes if interrupted
  python -m wspt.run_minatar leaderboard

Models come from config/models.toml in this repo. `api = "anthropic"` models go to the Claude API directly
(wspt/anthropic_direct.py); anything else goes through the bench's own OpenRouter client unchanged.
Results land in results/minatar/<model>/ (WSB_RESULTS is pointed here before the bench is imported).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("WSB_RESULTS", str(ROOT / "results"))
sys.path.insert(0, str(ROOT / "third_party" / "weight-space-bench"))

from weight_space_bench import config as C  # noqa: E402
from weight_space_bench.llm import OpenRouter  # noqa: E402
from weight_space_bench.minatar import leaderboard, task  # noqa: E402

from wspt.anthropic_direct import AnthropicDirect  # noqa: E402

MODELS = ROOT / "config" / "models.toml"


def client_for(model_cfg):
    if model_cfg.get("api") == "anthropic":
        return AnthropicDirect(model_cfg)
    if model_cfg.get("api") == "openai_compat":
        from wspt.openai_compat import OpenAICompat
        return OpenAICompat(model_cfg)
    return OpenRouter(model_cfg)


task.OpenRouter = client_for  # the only hook: task.ask() constructs its client through this name


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    cfg = C.load_minatar_config()
    if cmd == "run":
        names = sys.argv[2:] or [m["name"] for m in C.load_models(MODELS)]
        for n in names:
            task.run_model(C.model_by_name(n, MODELS), cfg)
    elif cmd == "reference":
        task.reference(cfg)
    elif cmd == "leaderboard":
        leaderboard.main()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
