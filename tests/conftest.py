"""Pytest configuration and shared fixtures."""
import os
import sys
from pathlib import Path

# faiss-cpu and torch each bundle their own libomp.dylib on macOS; loading
# both in the same process aborts ("Fatal Python error: Aborted") unless
# duplicate OpenMP runtimes are explicitly allowed. Must be set before
# faiss/torch are imported anywhere in the test session.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Add parent directory to path so we can import chunking modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock tiktoken to avoid downloading models during tests
# Tests will use fallback word-count token approximation instead
sys.modules['tiktoken'] = None  # Simulate ImportError


def pytest_configure(config):
    """Pytest hook: add custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (requires full corpus, use -m slow to run)"
    )
