"""
Configuration loading and validation utilities.

This module provides a function to load the application's configuration
from a YAML file and validate it against the Pydantic schema.
"""
from pathlib import Path
import yaml
from pydantic import ValidationError
from src.utils.config_schema import AppConfig


def load_config(config_path: Path) -> AppConfig:
    """
    Loads and validates the YAML configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        An AppConfig object with the validated configuration.
    """
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return AppConfig(**config_data)
