from typing import Optional, Dict, Any


class BaseConfig:
    """Базовый класс для конфигурации компонента."""
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
        }