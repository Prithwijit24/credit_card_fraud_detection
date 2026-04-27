"""Compatibility shim for older libraries expecting distutils.version."""

from __future__ import annotations

from functools import total_ordering

from packaging.version import InvalidVersion, Version


@total_ordering
class _BaseVersion:
    def __init__(self, version: str) -> None:
        self.vstring = version
        try:
            self._version = Version(version)
        except InvalidVersion:
            self._version = Version("0")

    def __repr__(self) -> str:
        return self.vstring

    def __str__(self) -> str:
        return self.vstring

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._version == other._version

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._version < other._version


class LooseVersion(_BaseVersion):
    pass


class StrictVersion(_BaseVersion):
    pass


__all__ = ["LooseVersion", "StrictVersion"]
