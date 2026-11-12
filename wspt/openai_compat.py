"""OpenAI-compatible chat client (Tinker's /oai endpoint for Inkling) with the weight-space-bench client interface.

Config fields: api = "openai_compat", base_url, key_env, model, reasoning_effort, max_output_tokens.
Reasoning comes back separately (`reasoning_content`, Tinker's default separate_reasoning=true); only the answer text
is parsed for weights, as with the other clients. No tools are ever sent.
"""
import os
import time

from openai import OpenAI


class OpenAICompat:
    def __init__(self, model_cfg):
        self.m = model_cfg
        if model_cfg.get("tools") or model_cfg.get("plugins"):
            raise SystemExit(f"{model_cfg['model']}: tools are not allowed in this benchmark")
        self.effort = model_cfg.get("reasoning_effort")
        self.client = OpenAI(base_url=model_cfg["base_url"], api_key=os.environ[model_cfg["key_env"]],
                             timeout=7200.0, max_retries=8)

    def chat(self, messages):
        kw = dict(model=self.m["model"], messages=messages, max_tokens=self.m.get("max_output_tokens", 64000))
        if self.effort:
            kw["reasoning_effort"] = self.effort
        t0 = time.time()
        r = self.client.chat.completions.create(**kw)
        msg = r.choices[0].message
        u = r.usage
        price_in, price_out = self.m.get("price_in", 0.0), self.m.get("price_out", 0.0)
        cost = (u.prompt_tokens * price_in + u.completion_tokens * price_out) / 1e6
        return {"text": msg.content or "", "reasoning": getattr(msg, "reasoning_content", None),
                "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens},
                "audit": {"tool_free": not getattr(msg, "tool_calls", None), "found": []}, "effort": self.effort,
                "max_tokens": kw["max_tokens"], "discarded_not_tool_free": 0, "cost": round(cost, 4),
                "finish": r.choices[0].finish_reason, "provider": self.m["base_url"], "model_id": r.model,
                "s": round(time.time() - t0, 1)}
