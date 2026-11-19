"""The rule-based recommender runs with no network and no LLM, API or MCP packages.

A data pipeline can run `synthony-recommender -m rulebased` in a task with no internet
and `SYNTHONY_LLM_PROVIDER=none`. The test runs the CLI in a fresh interpreter in which
any socket connection fails and the optional packages cannot be imported.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BLOCKED_PACKAGES = ["openai", "anthropic", "fastapi", "uvicorn", "sqlalchemy", "mcp"]

SITECUSTOMIZE = textwrap.dedent(
    f"""
    import socket, sys

    class _NoNetwork(OSError):
        pass

    def _refuse(*args, **kwargs):
        raise _NoNetwork("network access is blocked in this test")

    socket.socket.connect = _refuse
    socket.socket.connect_ex = _refuse
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse

    for name in {BLOCKED_PACKAGES!r}:
        sys.modules[name] = None  # makes `import name` raise ImportError
    """
)


@pytest.mark.functional
def test_rulebased_recommender_runs_offline_without_llm_packages(tmp_path: Path):
    rng = np.random.default_rng(0)
    data = tmp_path / "masked.parquet"
    pd.DataFrame(
        {
            "age": rng.integers(18, 90, 300),
            "income": rng.lognormal(10, 1.5, 300),
            "naics": rng.choice(["111110", "541511", "999999"], 300),
        }
    ).to_parquet(data)

    guard = tmp_path / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(SITECUSTOMIZE)

    out = tmp_path / "rec.json"
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(guard), os.environ.get("PYTHONPATH")])),
        "SYNTHONY_LLM_PROVIDER": "none",
    }
    for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "VLLM_URL", "SGLANG_URL"]:
        env.pop(key, None)

    proc = subprocess.run(
        [
            sys.executable, "-c",
            "import sys; from synthony.cli import app; app(sys.argv[1:])",
            "recommend", "-i", str(data), "-m", "rulebased", "-n", "15", "--cpu-only",
            "-o", str(out),
        ],
        env=env, cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    rec = json.loads(out.read_text())
    assert rec["method"].startswith("rule_based")  # the engine reports a versioned name
    assert rec["primary_recommendation"]["model_name"]
