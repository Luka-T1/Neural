import pytest

from neural.core.config import DEFAULT_CONFIG_PATH, Config
from neural.core.seed import subsystem_rng


@pytest.fixture
def rng():
    return subsystem_rng(0, "test")


@pytest.fixture
def base_config():
    return Config.from_dict({"seed": 0})


@pytest.fixture
def default_config_path():
    return DEFAULT_CONFIG_PATH
