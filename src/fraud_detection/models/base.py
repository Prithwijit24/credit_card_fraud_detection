from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    production_status: str
    notes: str


class BaseFraudModel(ABC):
    name: str
    role: str
    production_status: str = "available"
    notes: str = ""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            name=self.name,
            role=self.role,
            production_status=self.production_status,
            notes=self.notes,
        )

    def metadata(self) -> dict[str, str]:
        return asdict(self.spec)

    @abstractmethod
    def build(self, **params: Any) -> Any:
        raise NotImplementedError
