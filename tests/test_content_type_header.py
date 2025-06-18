import json

import requests


def test_form_urlencoded_content_type_error(api_base_url, auth_headers, model, basic_messages):
    """
    Test that using application/x-www-form-urlencoded Content-Type header
    results in a 400 error.
    """
    url = f"{api_base_url}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
    }

    # Make the API request with application/x-www-form-urlencoded Content-Type header
    headers = {**auth_headers, "Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Assert a 400 Bad Request response
    assert response.status_code == 400, (
        f"Expected status code 400 for form-urlencoded Content-Type, got {response.status_code} "
        f"in {response.text}"
    )

    # Verify error response structure
    response_body = response.json()
    assert "error" in response_body or "title" in response_body, (
        "Response should contain error information"
    )

    # Print error for debugging
    print(f"Error response for form-urlencoded Content-Type: {response_body}")


# Add tests for the compatibility endpoint
def test_compat_form_urlencoded_content_type_error(
    api_base_url, auth_headers, model, basic_messages
):
    """Test that using application/x-www-form-urlencoded Content-Type header in compat API
    results in a 400 error."""
    url = f"{api_base_url}/compat/v1/chat/completions"

    payload = {
        "model": model,
        "messages": basic_messages,
    }

    # Make the API request with application/x-www-form-urlencoded Content-Type header
    headers = {**auth_headers, "Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Assert a 400 Bad Request response
    assert response.status_code == 400, (
        f"Expected status code 400 for form-urlencoded Content-Type in compat API, "
        f"got {response.status_code} in {response.text}"
    )

    # Verify error response structure
    response_body = response.json()
    assert "error" in response_body or "title" in response_body, (
        "Response should contain error information"
    )

    # Print error for debugging
    print(f"Error response for form-urlencoded Content-Type in compat API: {response_body}")
