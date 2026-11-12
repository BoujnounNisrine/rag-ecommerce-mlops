import copy

import pytest

from src.config import ConfigError, load_params, validate


def test_repo_params_are_valid():
    load_params()  # lève une exception si params.yaml est invalide


def test_overlap_larger_than_size_is_rejected():
    params = copy.deepcopy(load_params())
    params["chunking"]["chunk_overlap"] = params["chunking"]["chunk_size"]
    with pytest.raises(ConfigError):
        validate(params)


def test_invalid_experiment_is_rejected():
    params = copy.deepcopy(load_params())
    params["experiments"].append({"name": "bad", "chunk_size": 0, "chunk_overlap": 0})
    with pytest.raises(ConfigError):
        validate(params)


def test_missing_section_is_rejected():
    params = copy.deepcopy(load_params())
    del params["chunking"]
    with pytest.raises(ConfigError):
        validate(params)
