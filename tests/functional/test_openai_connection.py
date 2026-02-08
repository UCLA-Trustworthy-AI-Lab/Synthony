#!/usr/bin/env python3
"""Simple test script to verify OpenAI API connection using .env credentials."""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Get OpenAI configuration from environment
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_URL", "https://api.openai.com/v1")
model = os.getenv("OPENAI_MODEL", "gpt-4o")

if not api_key:
    print("Error: OPENAI_API_KEY not found in .env file")
    print("Please add: OPENAI_API_KEY=sk-your-key-here")
    exit(1)

print(f"Connecting to: {base_url}")
print(f"Using model: {model}")
print("-" * 50)

# Create OpenAI client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Send a simple test message
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        max_tokens=50
    )

    print("Connection successful!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"Connection failed: {e}")
    exit(1)
