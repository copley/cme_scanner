import os
import yaml


def load_config(config_file: str = 'config.yaml') -> dict:
    """Load and return configuration from a YAML file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
