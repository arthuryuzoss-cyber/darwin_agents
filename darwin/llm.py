"""Provedores de IA intercambiáveis.

- MockProvider: determinístico, para testes (custo zero).
- OllamaProvider: modelo LOCAL e GRATUITO (https://ollama.com). Não precisa de API paga.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LLMResult:
    text: str
    cost_cents: int

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError("text deve ser str")
        if isinstance(self.cost_cents, bool) or not isinstance(self.cost_cents, int) or self.cost_cents < 0:
            raise ValueError("cost_cents deve ser int >= 0")


class LLMProvider(ABC):
    name = "base"

    @abstractmethod
    def estimate_cost_cents(self, prompt: str) -> int: ...

    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> LLMResult: ...


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, reply: str = "Conteúdo de exemplo gerado pelo mock.", cost_cents: int = 0,
                 fail_times: int = 0, fn: Callable[[str, str], str] | None = None):
        self.reply = reply
        self.cost_cents = cost_cents
        self.fail_times = fail_times
        self.fn = fn
        self.calls = 0
        self.last_system = ""

    def estimate_cost_cents(self, prompt: str) -> int:
        return self.cost_cents

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> LLMResult:
        self.calls += 1
        self.last_system = system
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("falha simulada do provedor")
        text = self.fn(system, prompt) if self.fn else self.reply
        return LLMResult(text, self.cost_cents)


class OllamaProvider(LLMProvider):
    """Usa um modelo local via Ollama (grátis). Por padrão só aceita localhost."""
    name = "ollama"

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434",
                 timeout_s: int = 180, allow_remote: bool = False):
        parsed = urllib.parse.urlparse(host)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("host inválido")
        if not allow_remote and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("host remoto bloqueado (use allow_remote=True se for intencional)")
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s

    def estimate_cost_cents(self, prompt: str) -> int:
        return 0

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> LLMResult:
        body = json.dumps({
            "model": self.model, "system": system, "prompt": prompt, "stream": False,
            "options": {"num_predict": max_tokens},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise RuntimeError(f"ollama indisponível: {type(e).__name__}") from None
        text = data.get("response")
        if not isinstance(text, str):
            raise RuntimeError("resposta inesperada do ollama")
        return LLMResult(text, 0)
