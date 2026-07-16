"""Conversão por idioma e mapeamento de estilo ASS."""
from __future__ import annotations

import admin_store as store
import main as app_main


def test_conversion_by_locale_returns_list():
    rows = store.conversion_by_locale(14)
    assert isinstance(rows, list)
    for row in rows:
        assert "locale" in row
        assert "visitas" in row
        assert "transcricoes" in row
        assert "taxa_conversao_pct" in row


def test_style_ass_maps_max_width_and_custom():
    style = {
        "fontSize": 40,
        "color": "#ffffff",
        "outline": 2,
        "shadow": "soft",
        "bg": True,
        "bgOpacity": 0.35,
        "align": "center",
        "position": "custom",
        "marginV": 48,
        "padding": 16,
        "maxWidthPct": 70,
    }
    ass = app_main._style_json_to_ass_force_style(style, 1280, 720)
    assert "Alignment=5" in ass
    assert "MarginL=" in ass
    assert "MarginR=" in ass


def test_normalize_ui_locale():
    assert app_main._normalize_ui_locale("en") == "en"
    assert app_main._normalize_ui_locale("PT") == "pt"
    assert app_main._normalize_ui_locale("xx") is None
