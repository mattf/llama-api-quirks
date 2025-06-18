"""
Tests for Llama API compatibility with LangChain using the standard test classes.
This file verifies that the Llama API's compatibility endpoint works with LangChain's
standard test classes.
"""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_tests.integration_tests import ChatModelIntegrationTests


@pytest.mark.asyncio
class TestLangChainStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatOpenAI

    @property
    def chat_model_params(self) -> dict:
        # Get the API key from environment
        import os

        api_key = os.environ.get("LLAMA_API_KEY")
        if api_key is None:
            pytest.skip("LLAMA_API_KEY environment variable not set")

        model = os.environ.get("LLAMA_MODEL", "Llama-3.3-8B-Instruct")

        return {
            "model": model,
            "base_url": "https://api.llama.com/compat/v1",
            "api_key": api_key,
        }
