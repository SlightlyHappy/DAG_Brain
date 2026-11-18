# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
import json
from typing import Dict, Any, List, Optional
import httpx
from .net import client as http_client
from .config import get_runtime_settings, PROVIDER_PRESETS


async def fetch_ollama_models(base_url: Optional[str] = None) -> List[str]:
    settings = get_runtime_settings()
    url = (base_url or settings.get("ollama_base_url") or "http://localhost:11434").rstrip("/")
    try:
        async with http_client(timeout=3.0) as client:
            resp = await client.get(f"{url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        pass
    return []


async def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Calls the configured LLM provider (OpenRouter, Ollama, Gemini, OpenAI, Anthropic, Custom)
    and parses a structured JSON response. Returns None if provider is unconfigured or unreachable
    so the caller can fall back to the built-in heuristic causal engine.
    """
    settings = get_runtime_settings()
    provider = provider_override or settings.get("active_provider", "openrouter")
    model = model_override or settings.get("active_model") or PROVIDER_PRESETS.get(provider, {}).get("default_model", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        async with http_client(timeout=45.0) as client:
            if provider == "ollama":
                base_url = (settings.get("ollama_base_url") or "http://localhost:11434").rstrip("/")
                payload = {
                    "model": model or "llama3.2",
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.3},
                }
                resp = await client.post(f"{base_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "")
                    return _extract_json(content)
                return None

            if provider == "anthropic":
                api_key = settings.get("anthropic_api_key", "").strip()
                if not api_key:
                    return None
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                payload = {
                    "model": model or "claude-3-7-sonnet-latest",
                    "max_tokens": 2500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "temperature": 0.3,
                }
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload,
                )
                if resp.status_code == 200:
                    blocks = resp.json().get("content", [])
                    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                    return _extract_json(text)
                return None

            # OpenAI-compatible endpoints: OpenRouter, Gemini, OpenAI, Custom
            if provider == "openrouter":
                api_key = settings.get("openrouter_api_key", "").strip()
                base_url = (settings.get("openrouter_base_url") or "https://openrouter.ai/api/v1").rstrip("/")
                if not api_key:
                    return None
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer": "https://github.com/dag-brain/dag-brain",
                    "X-Title": "DAG_Brain Causal Studio",
                    "Content-Type": "application/json",
                }
            elif provider == "gemini":
                api_key = settings.get("gemini_api_key", "").strip()
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
                if not api_key:
                    return None
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            elif provider == "openai":
                api_key = settings.get("openai_api_key", "").strip()
                base_url = "https://api.openai.com/v1"
                if not api_key:
                    return None
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
            else:
                # custom
                api_key = settings.get("custom_api_key", "lm-studio").strip()
                base_url = (settings.get("custom_base_url") or "http://localhost:1234/v1").rstrip("/")
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }

            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.3,
            }
            resp = await client.post(
                f"{base_url}/chat/completions", headers=headers, json=payload
            )
            if resp.status_code == 200:
                choices = resp.json().get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return _extract_json(content)
    except Exception:
        return None

    return None


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return None
    return None
