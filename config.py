"""
Configuration System - Manage application settings
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Application configuration."""

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    DEBUG: bool = False
    THREADED: bool = True

    # Model settings
    MODEL_PATH: str = "models/mnist_model.h5"
    MODEL_CACHE_SIZE: int = 100

    # Prediction settings
    MAX_IMAGE_SIZE_MB: float = 5.0
    BATCH_SIZE: int = 32
    PREDICTION_TIMEOUT_SECONDS: float = 30.0

    # Cache settings
    ENABLE_PREDICTION_CACHE: bool = True
    CACHE_MAX_SIZE: int = 100

    # Logging settings
    ENABLE_REQUEST_LOGGING: bool = True
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"

    # Database settings (future)
    DATABASE_URL: str = ""
    SAVE_PREDICTIONS: bool = True

    # API settings
    MAX_BATCH_SIZE: int = 100
    ENABLE_SWAGGER: bool = True

    @classmethod
    def from_env(cls):
        """Load config from environment variables."""
        return cls(
            HOST=os.getenv('HOST', cls.HOST),
            PORT=int(os.getenv('PORT', cls.PORT)),
            DEBUG=os.getenv('DEBUG', str(cls.DEBUG)).lower() == 'true',
            MODEL_PATH=os.getenv('MODEL_PATH', cls.MODEL_PATH),
            LOG_LEVEL=os.getenv('LOG_LEVEL', cls.LOG_LEVEL),
            ENABLE_PREDICTION_CACHE=os.getenv('ENABLE_CACHE', str(cls.ENABLE_PREDICTION_CACHE)).lower() == 'true',
        )

    @classmethod
    def from_file(cls, filepath: str):
        """Load config from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def save_to_file(self, filepath: str = "config.json"):
        """Save config to JSON file."""
        path = Path(filepath)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        print(f"Config saved to {path}")
        return path

    def to_dict(self):
        """Convert config to dictionary."""
        return asdict(self)

    def __str__(self):
        """String representation."""
        config_dict = asdict(self)
        lines = ["Configuration:"]
        for key, value in sorted(config_dict.items()):
            lines.append(f"  {key:<25} = {value}")
        return "\n".join(lines)


# Global config instance
_config = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        # Try to load from file, then environment, then use defaults
        if Path("config.json").exists():
            _config = Config.from_file("config.json")
        else:
            _config = Config.from_env()
    return _config


def set_config(config: Config):
    """Set global config instance."""
    global _config
    _config = config


# Example config file generator
def create_example_config(filepath: str = "config.example.json"):
    """Create example configuration file."""
    config = Config()
    config.save_to_file(filepath)
    print(f"Example config created at {filepath}")
