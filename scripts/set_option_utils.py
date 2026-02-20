#!/usr/bin/env python3
"""Shared configuration and helpers for set_option scripts."""

import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_OPTIONS = [
    "backward.isDefEq.respectTransparency",
    "backward.whnf.reducibleClassField",
]


def set_option_line(option: str) -> str:
    """Return the set_option line text for an option."""
    return f"set_option {option} false in\n"


def removable_pattern(option: str) -> re.Pattern:
    """Match bare `set_option X false in` lines (no trailing comment)."""
    escaped = re.escape(option)
    return re.compile(rf"^\s*set_option {escaped} false in\s*$")


def commented_pattern(option: str) -> re.Pattern:
    """Match `set_option X false in -- ...` lines."""
    escaped = re.escape(option)
    return re.compile(rf"^\s*set_option {escaped} false in\s+--")


def lakefile_pattern(option: str) -> re.Pattern:
    """Match lakefile entries for an option."""
    escaped = re.escape(option)
    return re.compile(
        rf"^\s*⟨`{escaped},\s*false⟩,?\s*\n",
        re.MULTILINE,
    )
