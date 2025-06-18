import json

import pytest
import requests


def test_chat_completions_basic_request(api_base_url, auth_headers, model, basic_messages):
    """Test basic chat completion functionality with the Llama API."""
    # Set up the request
    url = f"{api_base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
    }

    # Make the API request
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Assertions
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code} in {response.text}"
    )

    response_body = response.json()

    # Handle different API response formats
    if "choices" in response_body:
        # Standard OpenAI-compatible format
        assert len(response_body["choices"]) > 0, "Response should contain at least one choice"
        assert "message" in response_body["choices"][0], "Each choice should contain a message"
        assert "content" in response_body["choices"][0]["message"], "Message should contain content"
        assert response_body["choices"][0]["message"]["content"], "Content should not be empty"
    elif "completion_message" in response_body:
        # Custom format used by some Llama API providers
        assert "content" in response_body["completion_message"], "Response should contain content"
        assert "text" in response_body["completion_message"]["content"], (
            "Content should include text"
        )
        assert response_body["completion_message"]["content"]["text"], "Text should not be empty"
    else:
        pytest.fail(f"Unexpected response format: {response_body}")


def test_compat_chat_completions_basic_request(api_base_url, auth_headers, model, basic_messages):
    """Test basic chat completion functionality with the Llama API."""
    # Set up the request
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
    }

    # Make the API request
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Assertions
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code} in {response.text}"
    )

    response_body = response.json()
    assert "choices" in response_body, "Response should contain 'choices'"
    assert len(response_body["choices"]) > 0, "Response should contain at least one choice"
    assert "message" in response_body["choices"][0], "Each choice should contain a message"
    assert "content" in response_body["choices"][0]["message"], "Message should contain content"
    assert response_body["choices"][0]["message"]["content"], "Content should not be empty"
