from src.config import resolve
from src.data.cleaning import has_placeholder
from src.data.make_eval import in_scope_questions, out_of_scope_questions
from src.data.prepare import bitext_documents, policy_documents, split_markdown_sections


def test_bitext_documents_per_intent(bitext_df):
    docs = bitext_documents(bitext_df, responses_per_intent=3, min_words=5, seed=42)
    counts = {}
    for d in docs:
        counts[d["intent"]] = counts.get(d["intent"], 0) + 1
    assert counts == {"cancel_order": 3, "delivery_options": 3, "track_refund": 3}


def test_bitext_documents_are_reproducible(bitext_df):
    a = bitext_documents(bitext_df, 3, 5, seed=42)
    b = bitext_documents(bitext_df, 3, 5, seed=42)
    assert a == b


def test_bitext_documents_have_provenance_and_no_placeholder(bitext_df):
    for d in bitext_documents(bitext_df, 3, 5, seed=42):
        assert d["source"] == "bitext" and d["doc_id"] and d["intent"]
        assert not has_placeholder(d["text"])


def test_min_words_filter(bitext_df):
    assert bitext_documents(bitext_df, 3, min_words=10_000, seed=42) == []


def test_split_markdown_sections():
    md = "# Title\nintro\n## A\ntext a\n## B\n\n## C\ntext c\n"
    assert split_markdown_sections(md) == [("A", "text a"), ("C", "text c")]


def test_policy_documents_from_repo():
    docs = policy_documents(resolve("data/raw/policies"))
    assert len(docs) >= 5
    assert all(d["source"].startswith("policies/") and d["version"] for d in docs)
    # la contradiction volontaire doit être présente dans le corpus
    windows = [d["text"] for d in docs if d["title"] == "Return window"]
    assert any("30 days" in t for t in windows) and any("14 days" in t for t in windows)


def test_eval_questions_are_stratified(bitext_df):
    qs = in_scope_questions(bitext_df, per_intent=2, seed=42)
    assert len(qs) == 6
    assert {q["expected_intent"] for q in qs} == {"cancel_order", "track_refund", "delivery_options"}
    assert len({q["qid"] for q in qs}) == len(qs)


def test_out_of_scope_file_is_readable():
    qs = out_of_scope_questions(resolve("data/eval/out_of_scope.txt"))
    assert len(qs) >= 5
    assert all(not q["in_scope"] and not q["question"].startswith("#") for q in qs)
