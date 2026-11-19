"""The built wheel carries the data files that the package reads at run time.

An editable install reads them from the source tree, so only a regular install, such as
one in a container image, shows a missing file.
"""

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.skipif(
    importlib.util.find_spec("setuptools") is None,
    reason="building the wheel needs setuptools in this interpreter",
)
def test_wheel_contains_model_capabilities_json(tmp_path: Path):
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps",
         "--no-build-isolation", "-w", str(tmp_path), "-q"],
        check=True, cwd=tmp_path, capture_output=True, text=True,
    )
    (wheel,) = tmp_path.glob("synthony-*.whl")
    assert "synthony/recommender/model_capabilities.json" in zipfile.ZipFile(wheel).namelist()
