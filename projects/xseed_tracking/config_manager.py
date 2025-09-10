import os
import yaml
from dotenv import load_dotenv
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

    def load_config(self) -> Dict[str, Any]:
        """Load and process the configuration file."""
        load_dotenv()  # Load environment variables from .env file
        
        with open(self.config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self._expand_env_vars()
        self._validate_config()
        
        return self.config

    def _expand_env_vars(self) -> None:
        """Expand environment variables in the loaded config."""
        def expand(item: Any) -> Any:
            if isinstance(item, dict):
                return {k: expand(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [expand(i) for i in item]
            elif isinstance(item, str):
                return os.path.expandvars(item)
            else:
                return item

        self.config = expand(self.config)

    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Add any configuration validation logic here
        # For example, checking if required fields are present
        required_fields = ['data', 'paths', 'controlled_test']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required configuration field: {field}")

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration values."""
        return self.config[key]
