"""Pytest configuration and shared fixtures."""
import sys
from pathlib import Path

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
