#!/usr/bin/env python3
"""
Functional test for OpenAI API connection.

This test is skipped if OPENAI_API_KEY is not configured.
"""

import os
import pytest
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


@pytest.fixture
def openai_client():
    """Create OpenAI client if API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_URL", "https://api.openai.com/v1")
    
    if not api_key:
        pytest.skip("OPENAI_API_KEY not configured in .env")
    
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


class TestOpenAIConnection:
    """Tests for OpenAI API connectivity."""

    def test_openai_connection(self, openai_client):
        """Test basic OpenAI API connection with a simple prompt."""
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Say hello in one word."}
            ],
            max_tokens=10
        )
        
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    def test_openai_json_response(self, openai_client):
        """Test OpenAI API with JSON response format."""
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Return a JSON object with key 'status' set to 'ok'."}
            ],
            max_tokens=50,
            response_format={"type": "json_object"}
        )
        
        import json
        content = response.choices[0].message.content
        data = json.loads(content)
        assert "status" in data
