from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class OpenAICompatibleModel:
    """Tiny optional adapter for OpenAI-compatible chat completions APIs.

    It intentionally uses only the Python standard library so the package stays dependency-light.
    """

    model: str
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    timeout_seconds: int = 120

    def complete(self, prompt: str, *, role: str, metadata: Mapping[str, Any]) -> str:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAICompatibleModel")
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {"asi_role": role, **{k: str(v) for k, v in metadata.items()}},
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider returned HTTP {exc.code}: {body}") from exc
        return data["choices"][0]["message"]["content"]
