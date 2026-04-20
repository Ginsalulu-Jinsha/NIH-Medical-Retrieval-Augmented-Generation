from __future__ import annotations

import json, os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaConnectionError(RuntimeError):
    """Raised when the local Ollama server cannot be reached or returns an invalid response."""


@dataclass
class OllamaClient:
    host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    timeout_seconds: int = 120

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        request = Request(
            url=f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise OllamaConnectionError(
                f"Ollama returned HTTP {exc.code}. Make sure the model '{model}' is available."
            ) from exc
        except URLError as exc:
            raise OllamaConnectionError(
                "Could not connect to Ollama. Start it with `ollama serve` and make sure it is "
                "reachable at http://127.0.0.1:11434."
            ) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaConnectionError("Ollama returned a non-JSON response.") from exc

        if "response" not in parsed:
            raise OllamaConnectionError("Ollama response did not include a `response` field.")

        return str(parsed["response"]).strip()
