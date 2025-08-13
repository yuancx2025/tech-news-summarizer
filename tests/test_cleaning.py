import os
from src.cleaning import normalize_url, clean_text, content_hash, validate_row

def test_normalize_url_strips_tracking():
    u = "https://example.com/a?utm_source=x&gclid=123&keep=1#frag"
    v = normalize_url(u)
    assert "utm_source" not in v and "gclid" not in v and "#frag" not in v
    assert "keep=1" in v

def test_clean_text_collapses_whitespace_and_unidecodes():
    raw = "Hello\t\tworld…\n\n\nNew   line — test"
    out = clean_text(raw)
    assert "  " not in out  # no double spaces
    assert "\n\n\n" not in out
    assert "..." in out or "..." == out[-3:] or "-" in out  # unicode folded

def test_content_hash_stable_on_equivalent_text():
    a = "Hello—world"
    b = "Hello-world"  # after clean_text + unidecode they should be equivalent
    # Note: content_hash should be computed AFTER clean_text in pipeline
    assert content_hash(clean_text(a)) == content_hash(clean_text(b))

def test_validate_row_rules():
    ok = {"title": "A", "text": "x" * 400}
    bad1 = {"title": "", "text": "x" * 400}
    bad2 = {"title": "A", "text": "short"}
    assert validate_row(ok) == []
    assert "title" in validate_row(bad1)
    assert any(p.startswith("text<") for p in validate_row(bad2))
