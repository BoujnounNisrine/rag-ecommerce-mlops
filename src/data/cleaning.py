"""Nettoyage du texte Bitext.

Bitext contient des placeholders du type {{Order Number}}. On remplace ceux qui
ont un sens pour notre boutique fictive « ShopFès » par une valeur fixe, et les
autres par une formulation neutre. Chaque remplacement est déterministe, ce qui
garantit la reproductibilité du corpus.
"""

import re

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")

# Valeurs fictives de la boutique ShopFès
KNOWN_VALUES = {
    "Customer Support Phone Number": "+212 5 35 00 00 00",
    "Customer Support Email": "support@shopfes.example",
    "Customer Support Hours": "Monday to Saturday, 9am-6pm",
    "Website URL": "www.shopfes.example",
    "Live Chat Support": "the live chat on our website",
    "Online Company Portal Info": "your ShopFès account",
    "Online Customer Support Channel": "the Help Center",
    "Online Order Interaction": "Order History",
    "Online Navigation Step": "the 'My Account' section",
    "Company Name": "ShopFès",
}

GENERIC_REPLACEMENT = "the relevant information"


def replace_placeholders(text: str) -> str:
    def _sub(match: re.Match) -> str:
        return KNOWN_VALUES.get(match.group(1).strip(), GENERIC_REPLACEMENT)

    return PLACEHOLDER_RE.sub(_sub, text)


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return normalize_whitespace(replace_placeholders(text))


def has_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_RE.search(text))
