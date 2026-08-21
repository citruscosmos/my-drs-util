#!/usr/bin/env python3
"""OpenAI互換 chat completions クライアント。Ollamaでもオンラインエンドポイントでも同じコードパス。

Ollama:   --llm-base-url http://localhost:11434/v1 --llm-model gemma3:27b (api-key不要)
オンライン: --llm-base-url https://api.openai.com/v1 --llm-api-key-env OPENAI_API_KEY --llm-model gpt-4o
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import requests


class ChatClient:
    def __init__(self, base_url: str, model: str, api_key_env: str | None = None, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env) if api_key_env else None
        self.timeout = timeout

    def chat_vision(
        self, system_prompt: str, user_prompt: str, image_paths: list[str],
        max_tokens: int = 2048, temperature: float = 0.0,
    ) -> str:
        content = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            b64 = base64.b64encode(Path(p).read_bytes()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        r = requests.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
