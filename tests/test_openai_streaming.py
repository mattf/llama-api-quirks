import pytest
from openai import OpenAI


def test_compat_openai_sdk_streaming(api_base_url, api_key, model, basic_messages):
    """Test streaming functionality using the OpenAI Python SDK with compat endpoint."""
    # Create OpenAI client with compatibility endpoint URL
    client = OpenAI(api_key=api_key, base_url=f"{api_base_url}/compat/v1")

    errors = []
    chunks_received = 0

    try:
        # Request streaming response
        stream = client.chat.completions.create(model=model, messages=basic_messages, stream=True)

        # Process the stream
        all_content = ""
        for chunk in stream:
            chunks_received += 1

            # Validate chunk structure
            if not hasattr(chunk, "choices") or len(chunk.choices) == 0:
                errors.append(f"Chunk #{chunks_received}: Missing or empty choices array")
                continue

            # Check if delta exists and contains content
            if not hasattr(chunk.choices[0], "delta"):
                errors.append(f"Chunk #{chunks_received}: Choice doesn't contain delta")

            # Accumulate content if available
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                all_content += chunk.choices[0].delta.content

        # Print summary of received content
        if all_content:
            print(
                f"Received content: {all_content[:50]}..." if len(all_content) > 50 else all_content
            )

    except Exception as e:
        errors.append(f"Exception during streaming: {str(e)}")

    # Make sure we received at least one chunk
    if chunks_received == 0:
        errors.append("No streaming chunks received")
    else:
        print(f"Received {chunks_received} streaming chunks via OpenAI SDK with compat endpoint")

    # Report all errors at once
    if errors:
        pytest.fail("\n".join(errors))
