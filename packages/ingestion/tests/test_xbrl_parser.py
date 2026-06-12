"""Unit tests for the XBRL parser.

All tests are pure-Python with no DB or network calls.
"""

from __future__ import annotations

from mia_ingestion.edgar.xbrl_parser import CONCEPTS_OF_INTEREST, XBRLParser

# ── Sample EDGAR companyfacts payload ────────────────────────────────────────

SAMPLE_FACTS: dict = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "label": "Revenues",
                "units": {
                    "USD": [
                        # Instant (no 'start')
                        {
                            "end": "2023-01-29",
                            "val": 60922000000,
                            "accn": "0001045810-23-000017",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-24",
                            "frame": "CY2022",
                        },
                        # Duration (has 'start')
                        {
                            "start": "2022-01-31",
                            "end": "2023-01-29",
                            "val": 26914000000,
                            "accn": "0001045810-23-000017",
                            "fy": 2023,
                            "fp": "FY",
                            "form": "10-K",
                            "filed": "2023-02-24",
                        },
                    ]
                },
            },
            "NetIncomeLoss": {
                "label": "Net Income (Loss)",
                "units": {
                    "USD": [
                        {
                            "start": "2022-01-31",
                            "end": "2023-01-29",
                            "val": 4368000000,
                            "form": "10-K",
                            "filed": "2023-02-24",
                        }
                    ]
                },
            },
            # This concept is NOT in CONCEPTS_OF_INTEREST — should be filtered out
            "SomeObscureAccrualConcept": {
                "label": "Not tracked",
                "units": {"USD": [{"end": "2023-01-29", "val": 999}]},
            },
        }
    }
}


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_parse_only_returns_tracked_concepts() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    concepts = {f.concept for f in facts}
    assert "SomeObscureAccrualConcept" not in concepts
    assert "Revenues" in concepts
    assert "NetIncomeLoss" in concepts


def test_parse_revenue_entry_count() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    revenue_facts = [f for f in facts if f.concept == "Revenues"]
    assert len(revenue_facts) == 2


def test_parse_revenue_values() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    values = {f.value for f in facts if f.concept == "Revenues"}
    assert 60922000000.0 in values
    assert 26914000000.0 in values


def test_period_type_instant_vs_duration() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    revenue_facts = [f for f in facts if f.concept == "Revenues"]
    period_types = {f.period_type for f in revenue_facts}
    assert "instant" in period_types
    assert "duration" in period_types


def test_duration_has_period_start() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    duration_facts = [f for f in facts if f.period_type == "duration"]
    for f in duration_facts:
        assert f.period_start is not None


def test_instant_has_no_period_start() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    instant_facts = [f for f in facts if f.period_type == "instant"]
    for f in instant_facts:
        assert f.period_start is None


def test_ticker_and_cik_propagated() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    for fact in facts:
        assert fact.ticker == "NVDA"
        assert fact.cik == "0001045810"


def test_form_field_preserved() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    forms = {f.form for f in facts if f.form is not None}
    assert "10-K" in forms


def test_frame_field_preserved() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    frames = {f.frame for f in facts if f.frame is not None}
    assert "CY2022" in frames


def test_parse_empty_facts() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    assert parser.parse({"facts": {}}) == []
    assert parser.parse({}) == []


def test_null_value_entry_skipped() -> None:
    sample = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "label": "Revenues",
                    "units": {"USD": [{"end": "2023-01-29", "val": None}]},
                }
            }
        }
    }
    parser = XBRLParser(ticker="TEST", cik="0000000001")
    assert parser.parse(sample) == []


def test_multiple_tickers_isolated() -> None:
    """Parser for one ticker should not bleed into another."""
    p1 = XBRLParser(ticker="NVDA", cik="0001045810")
    p2 = XBRLParser(ticker="AAPL", cik="0000320193")
    facts1 = p1.parse(SAMPLE_FACTS)
    facts2 = p2.parse(SAMPLE_FACTS)
    assert all(f.ticker == "NVDA" for f in facts1)
    assert all(f.ticker == "AAPL" for f in facts2)


def test_concepts_of_interest_is_nonempty() -> None:
    assert len(CONCEPTS_OF_INTEREST) >= 20


def test_unit_field_set() -> None:
    parser = XBRLParser(ticker="NVDA", cik="0001045810")
    facts = parser.parse(SAMPLE_FACTS)
    for fact in facts:
        assert fact.unit == "USD"
