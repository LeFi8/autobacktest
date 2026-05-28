import pytest

from autobacktest.config import settings


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment() -> None:
    """Configure low-latency settings overrides for unit testing."""
    settings.sandbox_timeout = 2
