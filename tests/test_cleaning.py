from src.data.cleaning import clean_text, has_placeholder, replace_placeholders


def test_known_placeholder_is_replaced():
    out = replace_placeholders("Call {{Customer Support Phone Number}} now")
    assert out == "Call +212 5 35 00 00 00 now"


def test_unknown_placeholder_gets_generic_value():
    out = replace_placeholders("Your order {{Order Number}} is late")
    assert "{{" not in out and "the relevant information" in out


def test_clean_text_leaves_no_placeholder(bitext_df):
    for text in bitext_df["response"]:
        assert not has_placeholder(clean_text(text))


def test_clean_text_handles_non_string():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_whitespace_is_normalized():
    assert clean_text("a   b\n\n\n\nc  ") == "a b\n\nc"
