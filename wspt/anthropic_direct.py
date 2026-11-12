"""Direct Anthropic Messages API client with the same interface as weight_space_bench.llm.OpenRouter.

Used instead of OpenRouter for models whose config has `api = "anthropic"`. Parity with the published
weight-space-bench Opus 5.5 row (OpenRouter, reasoning.effort = "high", max_tokens 64,000, no tools):
  - output_config.effort = model_cfg["reasoning_effort"]   (Opus 5.5 defaults to "medium", so it is always set)
  - thinking = adaptive (always on for Opus 5.5)
  - max_tokens = model_cfg["max_output_tokens"]; streamed, since 64k needs streaming to avoid HTTP timeouts
  - no tools, no server-side refusal fallback (a fallback would silently swap the model under test; a refusal is
    recorded as the outcome instead)
"""
import time

import anthropic

# USD per 1M tokens (Claude API, first-party). Cache reads/writes priced per the API's caching multipliers.
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-fable-5-1": (10.00, 50.00)}


class AnthropicDirect:
    def __init__(self, model_cfg):
        self.m = model_cfg
        if model_cfg.get("tools") or model_cfg.get("plugins"):
            raise SystemExit(f"{model_cfg['model']}: tools are not allowed in this benchmark")
        self.effort = model_cfg.get("reasoning_effort", "high")
        self.client = anthropic.Anthropic(max_retries=8, timeout=3600.0)  # reads ANTHROPIC_API_KEY

    def chat(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        kw = dict(model=self.m["model"], max_tokens=self.m.get("max_output_tokens", 64000), messages=convo,
                  thinking={"type": "adaptive"}, output_config={"effort": self.effort})
        if system:
            kw["system"] = system
        t0 = time.time()
        with self.client.messages.stream(**kw) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text")
        u = msg.usage
        usage = {"prompt_tokens": u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0),
                 "completion_tokens": u.output_tokens, "input_tokens": u.input_tokens,
                 "cache_read_input_tokens": u.cache_read_input_tokens or 0,
                 "cache_creation_input_tokens": u.cache_creation_input_tokens or 0}
        pin, pout = PRICES.get(self.m["model"], (0.0, 0.0))
        cost = (u.input_tokens * pin + (u.cache_read_input_tokens or 0) * pin * 0.1
                + (u.cache_creation_input_tokens or 0) * pin * 1.25 + u.output_tokens * pout) / 1e6
        audit = {"tool_free": not any(b.type in ("tool_use", "server_tool_use") for b in msg.content), "found": []}
        return {"text": text, "reasoning": None, "usage": usage, "audit": audit, "effort": self.effort,
                "max_tokens": kw["max_tokens"], "discarded_not_tool_free": 0, "cost": round(cost, 4),
                "finish": msg.stop_reason, "stop_details": getattr(msg, "stop_details", None) and str(msg.stop_details),
                "provider": "anthropic-direct", "model_id": msg.model, "s": round(time.time() - t0, 1)}
