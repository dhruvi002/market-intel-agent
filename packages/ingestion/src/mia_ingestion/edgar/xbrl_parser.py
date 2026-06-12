"""Parse EDGAR XBRL companyfacts JSON into XBRLFact ORM objects.

The EDGAR companyfacts endpoint returns a JSON blob shaped like:
  {
    "facts": {
      "us-gaap": {
        "Revenues": {
          "label": "Revenues",
          "units": {
            "USD": [
              {"end": "2023-01-29", "val": 60922000000, "form": "10-K", ...},
              ...
            ]
          }
        },
        ...
      },
      "dei": { ... }
    }
  }

We filter to a curated set of ~25 concepts covering the income statement,
balance sheet, and cash flow statement — enough to power the SQL Generator
agent and to quantify financials in citations.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from mia_ingestion.models import XBRLFact

logger = logging.getLogger(__name__)

# Curated set of XBRL concepts tracked in the warehouse.
# Covers ~80% of the financial metrics analysts care about.
CONCEPTS_OF_INTEREST: frozenset[str] = frozenset(
    {
        # ── Income statement ──────────────────────────────────────────────────
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "GrossProfit",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "ResearchAndDevelopmentExpense",
        "GeneralAndAdministrativeExpense",
        "SellingGeneralAndAdministrativeExpense",
        "CostOfRevenue",
        "OperatingExpenses",
        # ── Balance sheet ─────────────────────────────────────────────────────
        "Assets",
        "LiabilitiesAndStockholdersEquity",
        "CashAndCashEquivalentsAtCarryingValue",
        "LongTermDebt",
        "StockholdersEquity",
        "CommonStockSharesOutstanding",
        "RetainedEarningsAccumulatedDeficit",
        "Goodwill",
        # ── Cash flow statement ───────────────────────────────────────────────
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInInvestingActivities",
        "NetCashProvidedByUsedInFinancingActivities",
        "PaymentsToAcquirePropertyPlantAndEquipment",
        # ── Other frequently used ─────────────────────────────────────────────
        "IncomeTaxExpenseBenefit",
        "DepreciationAndAmortization",
        "InterestExpense",
        "DividendsCommonStock",
        "StockRepurchasedAndRetiredDuringPeriodValue",
    }
)


class XBRLParser:
    """Transforms an EDGAR companyfacts JSON payload into a list of XBRLFact objects.

    Usage::

        parser = XBRLParser(ticker="NVDA", cik="0001045810")
        facts = parser.parse(facts_json)   # list[XBRLFact]
    """

    def __init__(self, ticker: str, cik: str) -> None:
        self.ticker = ticker
        self.cik = cik

    def parse(self, facts_json: dict[str, Any]) -> list[XBRLFact]:
        """Parse the top-level companyfacts dict.  Returns a flat list of facts."""
        records: list[XBRLFact] = []
        taxonomies: dict[str, Any] = facts_json.get("facts", {})

        for taxonomy_name, concepts in taxonomies.items():
            for concept_name, concept_data in concepts.items():
                if concept_name not in CONCEPTS_OF_INTEREST:
                    continue
                label: str = concept_data.get("label", concept_name)
                units: dict[str, list[dict]] = concept_data.get("units", {})

                for unit_name, entries in units.items():
                    for entry in entries:
                        fact = self._build_fact(
                            taxonomy=taxonomy_name,
                            concept=concept_name,
                            label=label,
                            unit=unit_name,
                            entry=entry,
                        )
                        if fact is not None:
                            records.append(fact)

        logger.info(
            "Parsed %d XBRL facts for %s (filtered from %d tracked concepts)",
            len(records),
            self.ticker,
            len(CONCEPTS_OF_INTEREST),
        )
        return records

    def _build_fact(
        self,
        taxonomy: str,
        concept: str,
        label: str,
        unit: str,
        entry: dict[str, Any],
    ) -> Optional[XBRLFact]:
        """Build one XBRLFact from a single entry dict.  Returns None if val is missing."""
        raw_val = entry.get("val")
        if raw_val is None:
            return None

        end_str: str | None = entry.get("end")
        start_str: str | None = entry.get("start")
        period_end = _parse_date(end_str)
        period_start = _parse_date(start_str)
        # If there's a start date it's a duration (income/cash-flow); otherwise instant (balance)
        period_type = "duration" if start_str else "instant"

        return XBRLFact(
            ticker=self.ticker,
            cik=self.cik,
            taxonomy=taxonomy,
            concept=concept,
            label=label,
            value=float(raw_val),
            unit=unit,
            period_type=period_type,
            period_start=period_start,
            period_end=period_end,
            form=entry.get("form"),
            frame=entry.get("frame"),
        )


def _parse_date(s: str | None) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None
