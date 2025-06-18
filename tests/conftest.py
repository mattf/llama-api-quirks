import os

import pytest


@pytest.fixture
def api_base_url():
    """Fixture to provide the Llama API base URL."""
    return os.environ.get("LLAMA_API_BASE_URL", "https://api.llama.com")


@pytest.fixture
def api_key():
    """Fixture to provide the Llama API key."""
    api_key = os.environ.get("LLAMA_API_KEY", None)
    if api_key is None:
        pytest.skip("LLAMA_API_KEY environment variable not set")
    return api_key


@pytest.fixture
def auth_headers(api_key):
    """Fixture to provide authentication headers."""
    return {"Authorization": f"Bearer {api_key}"}


@pytest.fixture
def model():
    """Fixture to provide the model under test."""
    return os.environ.get("LLAMA_MODEL", "Llama-3.3-8B-Instruct")


@pytest.fixture
def basic_messages():
    """Fixture to provide a basic set of messages for chat completion."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
