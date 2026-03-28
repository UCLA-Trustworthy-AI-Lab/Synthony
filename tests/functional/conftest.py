"""
Conftest for functional tests.

Some files in this directory are manual live-server scripts (not pytest tests).
They require a running server at localhost:8000 and local dataset files.
They are excluded from automated CI collection via collect_ignore.
"""

collect_ignore = [
    "test_api_advanced.py",
    "test_llm_with_systemprompt.py",
]
