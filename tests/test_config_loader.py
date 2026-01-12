import os
import pytest
from src.utils.config_loader import ConfigLoader

def test_config_loader_reads_values():
    cfg_path = os.path.join(os.path.dirname(__file__), os.pardir, "config", "default.yaml")
    loader = ConfigLoader(cfg_path)

    # Example keys you expect to exist in your default.yaml
    max_seq_len = loader.get("model.max_seq_len")
    cnn_filters = loader.get("model.cnn_filters")

    assert isinstance(max_seq_len, int), "max_seq_len should be an integer"
    assert max_seq_len > 0, "max_seq_len should be positive"

    assert isinstance(cnn_filters, int), "cnn_filters should be an integer"
    assert cnn_filters > 0, "cnn_filters should be positive"

    # Also check a value you added, e.g., languages list
    languages = loader.get("languages")
    assert isinstance(languages, list), "languages should be a list"
    assert "en" in languages, "English ('en') should be in supported languages"
