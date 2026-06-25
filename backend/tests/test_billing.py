"""Testes unitários — billing e exportação DOCX."""
from __future__ import annotations

import os
from unittest.mock import patch

import billing


def test_pricing_hidden_on_by_default():
    with patch.object(billing, "_store") as store:
        store.return_value.get_config.return_value = {"pricing_hidden": "1"}
        with patch.dict(os.environ, {"PRICING_HIDDEN": "1"}, clear=False):
            assert billing.pricing_hidden() is True


def test_pricing_hidden_off_when_config_disabled():
    with patch.object(billing, "_store") as store:
        store.return_value.get_config.return_value = {"pricing_hidden": "0"}
        assert billing.pricing_hidden() is False


def test_billing_config_hides_prices_when_pricing_hidden():
    with patch.object(billing, "_store") as store:
        store.return_value.get_config.return_value = {
            "billing_enabled": "0",
            "pricing_hidden": "1",
            "pro_price_label": "9,99 €/mês",
            "pro_quota_daily": "200",
        }
        with patch.dict(os.environ, {"PRICING_HIDDEN": "1", "BILLING_ENABLED": "0"}, clear=False):
            cfg = billing.billing_config()
    assert cfg["pricing_hidden"] is True
    assert cfg["price_label"] == ""
    assert cfg["pro_quota_daily"] == 0
    assert cfg["checkout_ready"] is False


def test_pro_quota_limit_reads_config_not_public_mask():
    with patch.object(billing, "_store") as store:
        store.return_value.get_config.return_value = {"pro_quota_daily": "250"}
        assert billing.pro_quota_limit() == 250


def test_build_docx_bytes_creates_valid_file():
    data = billing.build_docx_bytes("Olá mundo.\nSegunda linha.", title="Teste")
    assert isinstance(data, bytes)
    assert len(data) > 100
    assert data[:2] == b"PK"  # ZIP/DOCX
