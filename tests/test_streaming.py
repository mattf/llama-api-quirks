"""
This file contains tests for streaming behavior with the Llama API.

Test Matrix:
1. Standard endpoint (/v1/chat/completions):
   - stream=True + Accept: text/event-stream header → Should stream (SSE response)
   - stream=True + no Accept header → Should stream (SSE response)
   - stream=False/unset + Accept: text/event-stream header → Should NOT stream (JSON response)
   - stream=False + no Accept header → Should NOT stream (JSON response)

2. Compatibility endpoint (/compat/v1/chat/completions):
   - stream=True + Accept: text/event-stream header → Should stream (SSE response)
   - stream=True + no Accept header → Should stream (SSE response)
   - stream=False + Accept: text/event-stream header → Should NOT stream (JSON response)
   - stream=False + no Accept header → Should NOT stream (JSON response)

These tests verify that:
- The 'stream' parameter is the primary driver of streaming behavior
- The API correctly handles the Accept header
- Streaming responses use proper SSE format with text/event-stream Content-Type
- Non-streaming responses return application/json Content-Type
"""

import json

import pytest
import requests


def test_streaming_with_accept_header(api_base_url, auth_headers, model, basic_messages):
    """Test streaming with explicit Accept: text/event-stream header."""
    url = f"{api_base_url}/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": True}

    # Make the API request with explicit Accept header for SSE
    headers = {**auth_headers, "Content-Type": "application/json", "Accept": "text/event-stream"}
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Content-Type header check
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "text/event-stream":
        errors.append(f"Expected 'text/event-stream' Content-Type, got {content_type}")

    # Read and validate streaming chunks
    chunks_received = 0

    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            chunks_received += 1
            if not chunk.startswith("data: "):
                errors.append(f"Chunk #{chunks_received} doesn't follow SSE format: {chunk}")

    # Make sure we received at least one chunk
    if chunks_received == 0:
        errors.append("No streaming chunks received")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_streaming_without_accept_header(api_base_url, auth_headers, model, basic_messages):
    """Test streaming without Accept header (only stream=True in payload)."""
    url = f"{api_base_url}/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": True}

    # Make the API request WITHOUT Accept header
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # Collect all assertion failures
    errors = []

    # Status code check - now expecting 400 instead of 200
    if response.status_code != 400:
        errors.append(f"Expected status code 400, got {response.status_code} in {response.text}")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_accept_header_without_streaming(api_base_url, auth_headers, model, basic_messages):
    """Test what happens when Accept: text/event-stream is set but stream=False."""
    url = f"{api_base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
        # No stream parameter (defaults to False)
    }

    # Make the API request with Accept header but no stream parameter
    headers = {**auth_headers, "Content-Type": "application/json", "Accept": "text/event-stream"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Check if response is JSON not SSE
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "application/json":
        errors.append(f"Expected 'application/json' Content-Type, got {content_type}")

    # Try to parse as JSON
    try:
        response.json()
    except json.JSONDecodeError:
        errors.append("Failed to parse response as JSON")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


# Also test the compatibility endpoint
def test_compat_streaming_with_accept_header(api_base_url, auth_headers, model, basic_messages):
    """Test compat endpoint streaming with explicit Accept: text/event-stream header."""
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": True}

    # Make the API request with explicit Accept header for SSE
    headers = {**auth_headers, "Content-Type": "application/json", "Accept": "text/event-stream"}
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Content-Type header check
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "text/event-stream":
        errors.append(f"Expected 'text/event-stream' Content-Type, got {content_type}")

    # Read and validate streaming chunks
    chunks_received = 0

    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            chunks_received += 1
            if not chunk.startswith("data: "):
                errors.append(f"Chunk #{chunks_received} doesn't follow SSE format: {chunk}")

    # Make sure we received at least one chunk
    if chunks_received == 0:
        errors.append("No streaming chunks received")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_no_streaming_without_accept_header(api_base_url, auth_headers, model, basic_messages):
    """Test non-streaming without Accept header (stream=False in payload)."""
    url = f"{api_base_url}/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": False}

    # Make the API request WITHOUT Accept header for non-streaming
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Content-Type header check - should be application/json for non-streaming
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "application/json":
        errors.append(f"Expected 'application/json' Content-Type, got {content_type}")

    # Try to parse response as JSON
    try:
        response.json()
    except json.JSONDecodeError:
        errors.append("Failed to parse response as JSON")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_compat_streaming_without_accept_header(api_base_url, auth_headers, model, basic_messages):
    """Test compat endpoint streaming without Accept header (only stream=True in payload)."""
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": True}

    # Make the API request WITHOUT Accept header
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload), stream=True)

    # Collect all assertion failures
    errors = []

    # Status code check - now expecting 400 instead of 200
    if response.status_code != 400:
        errors.append(f"Expected status code 400, got {response.status_code} in {response.text}")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_compat_accept_header_without_streaming(api_base_url, auth_headers, model, basic_messages):
    """
    Test what happens when Accept: text/event-stream is set
    but stream=False for compat endpoint.
    """
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
        "stream": False,  # Explicitly set to False
    }

    # Make the API request with Accept header but stream=False
    headers = {**auth_headers, "Content-Type": "application/json", "Accept": "text/event-stream"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Check if response is JSON not SSE
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "application/json":
        errors.append(f"Expected 'application/json' Content-Type, got {content_type}")

    # Try to parse as JSON
    try:
        response.json()
    except json.JSONDecodeError:
        errors.append("Failed to parse response as JSON")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))


def test_compat_no_streaming_without_accept_header(
    api_base_url, auth_headers, model, basic_messages
):
    """Test compat endpoint non-streaming without Accept header."""
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {"model": model, "messages": basic_messages, "stream": False}

    # Make the API request WITHOUT Accept header for non-streaming
    headers = {**auth_headers, "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Collect all assertion failures
    errors = []

    # Status code check
    if response.status_code != 200:
        errors.append(f"Expected status code 200, got {response.status_code} in {response.text}")

    # Content-Type header check - should be application/json for non-streaming
    content_type = response.headers.get("Content-Type", "").split(";")[0]
    if content_type != "application/json":
        errors.append(f"Expected 'application/json' Content-Type, got {content_type}")

    # Try to parse response as JSON
    try:
        response.json()
    except json.JSONDecodeError:
        errors.append("Failed to parse response as JSON")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))
