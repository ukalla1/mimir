"""
Loads and queries named landmark positions from a YAML config file.
"""
import os
import yaml


_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__), '..', 'config', 'landmarks.yaml'
)


class LandmarkLoader:
    def __init__(self, config_path: str = None):
        path = config_path or _DEFAULT_CONFIG
        with open(os.path.abspath(path), 'r') as f:
            data = yaml.safe_load(f)
        self._landmarks: dict = data.get('landmarks', {})

    def get(self, name: str) -> dict:
        """Return position dict for a landmark name. Raises KeyError if not found."""
        name = name.strip().lower()
        if name not in self._landmarks:
            available = ', '.join(self._landmarks.keys())
            raise KeyError(
                f"Landmark '{name}' not found. Available: {available}"
            )
        return self._landmarks[name]

    def list_all(self) -> list[dict]:
        """Return all landmarks as a list of {name, x, y, description}."""
        result = []
        for name, info in self._landmarks.items():
            result.append({'name': name, **info})
        return result
