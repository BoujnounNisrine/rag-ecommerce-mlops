"""Fixtures partagées : un mini-CSV au format Bitext, sans réseau."""

import pandas as pd
import pytest

INTENTS = [
    ("cancel_order", "ORDER"),
    ("track_refund", "REFUND"),
    ("delivery_options", "DELIVERY"),
]


@pytest.fixture
def bitext_df() -> pd.DataFrame:
    rows = []
    for intent, category in INTENTS:
        for i in range(8):
            rows.append(
                {
                    "flags": "BL",
                    "instruction": f"how do I {intent.replace('_', ' ')} number {i}",
                    "category": category,
                    "intent": intent,
                    "response": (
                        f"Response {i} for {intent}. Please contact {{{{Customer Support Phone Number}}}} "
                        f"or visit {{{{Website URL}}}} for help with {{{{Order Number}}}}. "
                        + "We are happy to assist you with your request today. " * 3
                    ),
                }
            )
    return pd.DataFrame(rows)
