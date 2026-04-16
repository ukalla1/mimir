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
        raw: dict = data.get('landmarks', {})
        # Store with original-case display names; build a lowercase lookup index
        # so callers can use any case (YAML has "Reception" but LLM often sends "reception").
        self._landmarks: dict = raw
        self._by_lower: dict = {k.strip().lower(): k for k in raw.keys()}

    def get(self, name: str) -> dict:
        """Return position dict for a landmark name (case-insensitive).
        Raises KeyError if not found."""
        key = name.strip().lower()
        orig = self._by_lower.get(key)
        if orig is None:
            available = ', '.join(self._landmarks.keys())
            raise KeyError(
                f"Landmark '{name}' not found. Available: {available}"
            )
        return self._landmarks[orig]

    def list_all(self) -> list[dict]:
        """Return all landmarks as a list of {name, x, y, description}."""
        result = []
        for name, info in self._landmarks.items():
            result.append({'name': name, **info})
        return result
