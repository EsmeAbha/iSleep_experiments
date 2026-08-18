"""Put `code/` on sys.path so the modules import under their own names.

The modules import each other flatly (`from features import ...`), which works
when a script is run from inside `code/` but not when pytest collects from the
repo root -- so the path is set up once here.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)


@pytest.fixture(scope="session")
def repo_root():
    return ROOT


@pytest.fixture(scope="session")
def results_dir():
    return os.path.join(ROOT, "results")


@pytest.fixture
def rng():
    """Seeded generator -- every test that makes data must be reproducible."""
    import numpy as np
    return np.random.default_rng(42)
