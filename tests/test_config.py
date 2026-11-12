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


@pytest.mark.parametrize(
    "section,key,value",
    [("retrieval", "top_k", 0), ("retrieval", "min_score", 1.5),
     ("generation", "temperature", -0.1), ("generation", "max_tokens", 0),
     ("generation", "reasoning_effort", "turbo")],
)
def test_invalid_rag_settings_are_rejected(section, key, value):
    params = copy.deepcopy(load_params())
    params[section][key] = value
    with pytest.raises(ConfigError):
        validate(params)


def test_missing_section_is_rejected():
    params = copy.deepcopy(load_params())
    del params["chunking"]
    with pytest.raises(ConfigError):
        validate(params)
