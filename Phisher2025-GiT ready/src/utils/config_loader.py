"""
Configuration loader for handling YAML config files.
"""
import yaml
from typing import Any, Optional


class ConfigLoader:
    """Load and access configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/default.yaml"):
        """
        Initialize ConfigLoader with a YAML config file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> dict:
        """Load YAML configuration file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.full_load(f)
                return config if config is not None else {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key using dot notation (e.g., 'model.max_seq_len')
            default: Default value if key not found
            
        Returns:
            Configuration value or default if not found
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_all(self) -> dict:
        """Get entire configuration dictionary."""
        return self.config
