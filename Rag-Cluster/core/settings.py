"""Configuration for the optional API-backed RAG examples.

No credential or private endpoint is distributed with the repository. Set the
environment variables below before running an API-backed script. The OpenAI
SDK-compatible endpoint may be replaced by any compatible provider.
"""

import os


class Settings:
    def __init__(self) -> None:
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.environ.get("OPENAI_TEMPERATURE", "0"))


def get_settings():
    return Settings()
