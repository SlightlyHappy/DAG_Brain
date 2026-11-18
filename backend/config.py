# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
import os
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = DATA_DIR / "runtime_settings.json"

PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "name": "OpenRouter (Unified Cloud AI)",
        "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "default_model": os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.5-flash"),
        "suggested_models": [
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-3.7-sonnet",
            "deepseek/deepseek-r1",
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "meta-llama/llama-3.3-70b-instruct",
            "qwen/qwen-2.5-72b-instruct",
        ],
    },
    "ollama": {
        "name": "Ollama (Local & Private Self-Hosted)",
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "default_model": os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2"),
        "suggested_models": [
            "llama3.2",
            "llama3.1:8b",
            "deepseek-r1:8b",
            "qwen2.5:14b",
            "mistral",
            "gemma2:9b",
        ],
    },
    "gemini": {
        "name": "Google Gemini (Direct API)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash"),
        "suggested_models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
    },
    "openai": {
        "name": "OpenAI (Direct API)",
        "base_url": "https://api.openai.com/v1",
        "default_model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini"),
        "suggested_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini",
        ],
    },
    "anthropic": {
        "name": "Anthropic Claude (Direct API)",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": os.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-3-7-sonnet-latest"),
        "suggested_models": [
            "claude-3-7-sonnet-latest",
            "claude-3-5-haiku-latest",
        ],
    },
    "custom": {
        "name": "Custom OpenAI-Compatible (LM Studio / vLLM / Institutional)",
        "base_url": os.getenv("CUSTOM_OPENAI_BASE_URL", "http://localhost:1234/v1"),
        "default_model": os.getenv("CUSTOM_OPENAI_MODEL", "local-model"),
        "suggested_models": ["local-model"],
    },
}


def get_runtime_settings() -> Dict[str, Any]:
    defaults = {
        "active_provider": os.getenv("DEFAULT_LLM_PROVIDER", "openrouter"),
        "active_model": os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.5-flash"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "openrouter_base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "custom_base_url": os.getenv("CUSTOM_OPENAI_BASE_URL", "http://localhost:1234/v1"),
        "custom_api_key": os.getenv("CUSTOM_OPENAI_API_KEY", "lm-studio"),
        "tavily_api_key": os.getenv("TAVILY_API_KEY", ""),
        "core_api_key": os.getenv("CORE_API_KEY", ""),
        "serpapi_api_key": os.getenv("SERPAPI_API_KEY", ""),
        "semantic_scholar_api_key": os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""),
        "searxng_base_url": os.getenv("SEARXNG_BASE_URL", ""),
        "ezproxy_prefix": os.getenv("EZPROXY_PREFIX", ""),
        "offline_mode": os.getenv("OFFLINE_MODE", "").strip().lower() in ("1", "true", "yes", "on"),
        "scholarly_email": os.getenv("SCHOLARLY_CONTACT_EMAIL", "researcher@dagbrain.org"),
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if v is not None and v != "":
                    defaults[k] = v
        except Exception:
            pass
    return defaults


def save_runtime_settings(updates: Dict[str, Any]) -> Dict[str, Any]:
    current = get_runtime_settings()
    for k, v in updates.items():
        if k in current and v is not None:
            current[k] = v
    SETTINGS_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    return current


def mask_settings_for_client(settings: Dict[str, Any]) -> Dict[str, Any]:
    masked = dict(settings)
    for key_field in [
        "openrouter_api_key",
        "gemini_api_key",
        "openai_api_key",
        "anthropic_api_key",
        "custom_api_key",
        "tavily_api_key",
        "core_api_key",
        "serpapi_api_key",
        "semantic_scholar_api_key",
    ]:
        val = masked.get(key_field, "")
        masked[f"{key_field}_configured"] = bool(val and len(val.strip()) > 3)
        if val and len(val) > 8:
            masked[key_field] = val[:4] + "..." + val[-4:]
        elif val:
            masked[key_field] = "****"
        else:
            masked[key_field] = ""
    masked["provider_presets"] = PROVIDER_PRESETS
    return masked
