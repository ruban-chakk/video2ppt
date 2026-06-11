from __future__ import annotations

import json
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .env import load_dotenv


@dataclass
class OllamaClient:
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout: int = 1800
    max_json_retries: int = 2

    def generate_json(self, system: str, user: str) -> dict[str, Any]:
        prompt = f"{system.strip()}\n\n{user.strip()}\n\nReturn only valid JSON."
        text = self._generate_text(prompt)
        try:
            return parse_json_response(text)
        except json.JSONDecodeError as exc:
            if self.max_json_retries <= 0:
                raise

            retry_text = self._generate_text(prompt, temperature=0.0)
            try:
                return parse_json_response(retry_text)
            except json.JSONDecodeError:
                pass

            repair_prompt = f"""
Repair this invalid JSON into valid JSON only.
Do not add new information. Do not include markdown.

Invalid JSON:
{text}
"""
            repaired = self._generate_text(repair_prompt, temperature=0.0)
            try:
                return parse_json_response(repaired)
            except json.JSONDecodeError as repaired_exc:
                raise RuntimeError(
                    f"Ollama returned invalid JSON after repair attempt: {repaired_exc}"
                ) from exc

    def _generate_text(self, prompt: str, temperature: float | None = None) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_ctx": 8192,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except socket.timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout} seconds while using {self.model}. "
                "Increase --llm-timeout or reduce --max-chunk-chars."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. Start Ollama and confirm the model is available."
            ) from exc

        text = str(payload.get("response") or payload.get("thinking") or "").strip()
        if not text:
            done_reason = payload.get("done_reason", "unknown")
            raise RuntimeError(
                f"Ollama returned an empty response for {self.model}; done_reason={done_reason}."
            )
        return text


@dataclass
class OpenAIClient:
    model: str = "gpt-5.5"
    api_key: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: int = 1800
    use_web_search: bool = True

    def generate_json(self, system: str, user: str) -> dict[str, Any]:
        load_dotenv()
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or export it in your shell.")

        body: dict[str, Any] = {
            "model": self.model,
            "input": f"{system.strip()}\n\n{user.strip()}\n\nReturn only valid JSON.",
        }
        if self.use_web_search:
            body["tools"] = [{"type": "web_search", "search_context_size": "low"}]
            body["tool_choice"] = "auto"

        req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=openai_ssl_context()) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except socket.timeout as exc:
            raise RuntimeError(
                f"OpenAI request timed out after {self.timeout} seconds while using {self.model}."
            ) from exc
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Could not reach OpenAI API at {self.base_url}: {exc.reason}"
            ) from exc

        text = extract_openai_text(payload).strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty response.")
        return parse_json_response(text)


def openai_ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def extract_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    chunks: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def parse_json_response(text: str) -> dict[str, Any]:
    text = strip_markdown_fence(text.strip())

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise


def strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text)
    return re.sub(r"\s*```$", "", text).strip()
