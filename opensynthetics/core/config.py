"""Configuration management for OpenSynthetics."""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

CONFIG_FILE_NAME = "opensynthetics_config.json"
DEFAULT_BASE_DIR = Path.home() / "opensynthetics_data"

class Config:
    """Manages OpenSynthetics configuration."""

    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings

    @property
    def environment(self) -> str:
        return self.settings.get("environment", "dev")

    @property
    def storage(self) -> Dict[str, Any]:
        storage_settings = self.settings.get("storage", {"base_dir": str(DEFAULT_BASE_DIR)})
        if "base_dir" in storage_settings and isinstance(storage_settings["base_dir"], Path):
            storage_settings["base_dir"] = str(storage_settings["base_dir"])
        return storage_settings

    @property
    def base_dir(self) -> Path:
        return Path(self.storage.get("base_dir", str(DEFAULT_BASE_DIR)))

    @property
    def api_keys(self) -> Dict[str, str]:
        return self.settings.get("api_keys", {})

    @property
    def jwt_secret_key(self) -> Optional[str]:
        return self.settings.get("jwt_secret_key", "default-secret-key-please-change")

    @property
    def default_openai_model(self) -> str:
        return self.settings.get("default_openai_model", "gpt-3.5-turbo")

    @property
    def default_embedding_model(self) -> str:
        return self.settings.get("default_embedding_model", "text-embedding-3-small")

    def get_api_key(self, provider: str) -> Optional[str]:
        return self.api_keys.get(provider)

    @classmethod
    def get_config_path(cls) -> Path:
        env_path = os.getenv("OPENSYNTHETICS_CONFIG_PATH")
        if env_path:
            return Path(env_path)
        
        app_dir = Path.home() / ".opensynthetics"
        app_dir.mkdir(parents=True, exist_ok=True)
        return app_dir / CONFIG_FILE_NAME

    @classmethod
    def load(cls) -> "Config":
        config_path = cls.get_config_path()
        settings = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    settings = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {config_path}. Using default config.")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}. Using default config.")
        else:
            print(f"Info: Config file not found at {config_path}. Using default config. You can create one or use 'opensynthetics config set'.")
        
        base_dir_path_obj = Path(settings.get("storage", {}).get("base_dir", str(DEFAULT_BASE_DIR)))
        base_dir_path_obj.mkdir(parents=True, exist_ok=True)
        
        current_storage_config = settings.setdefault("storage", {})
        current_storage_config["base_dir"] = str(base_dir_path_obj)

        return cls(settings)

    def save(self) -> None:
        config_path = self.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        serializable_settings = json.loads(json.dumps(self.settings, default=str))

        with open(config_path, "w") as f:
            json.dump(serializable_settings, f, indent=2)
        print(f"Configuration saved to {config_path}")

    def set_value(self, key_path: str, value: Any) -> None:
        keys = key_path.split('.')
        current_level = self.settings
        for i, key in enumerate(keys):
            if i == len(keys) - 1:
                current_level[key] = value
            else:
                current_level = current_level.setdefault(key, {})
        self.save()

    def get_value(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        current_level = self.settings
        for key in keys:
            if isinstance(current_level, dict) and key in current_level:
                current_level = current_level[key]
            else:
                return default
        return current_level

if __name__ == "__main__":
    config = Config.load()
    print(f"Current base directory: {config.base_dir}")
    
    # config.set_value("api_keys.openai", "YOUR_OPENAI_KEY_HERE")
    # print(f"OpenAI API Key: {config.get_api_key('openai')}")

    # config.set_value("jwt_secret_key", "a-much-stronger-secret-key-than-this")
    # print(f"JWT Secret Key: {config.jwt_secret_key}")

    # new_storage_path = Path.home() / "my_opensyn_data_custom"
    # config.set_value("storage.base_dir", str(new_storage_path))
    # print(f"Updated base directory: {config.base_dir}")
    
    # assert new_storage_path.exists(), "Custom storage directory was not created."
    # print("Test finished.")