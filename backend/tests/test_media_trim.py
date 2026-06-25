"""Testes — parâmetros de corte de media."""
from __future__ import annotations

from main import _parse_trim_form


def test_parse_trim_accepts_valid_range():
    start, end = _parse_trim_form("10.5", "125.0")
    assert start == 10.5
    assert end == 125.0


def test_parse_trim_rejects_invalid_range():
    assert _parse_trim_form("50", "10") == (None, None)
    assert _parse_trim_form(None, None) == (None, None)


def test_parse_trim_start_only():
    start, end = _parse_trim_form("0", None)
    assert start == 0.0
    assert end is None
