"""trust/secret_scan.py — the credential scanner behind the gate's secret-argument warning."""

from trust import secret_scan


def test_detects_common_secret_kinds():
    text = (
        "anthropic sk-ant-abcdefghij1234567890ABCD, "
        "openai sk-proj-abcdefghij1234567890ABCD, "
        "tavily tvly-abcdefghij1234567890, "
        "email a.b@example.com, "
        "bearer Bearer abcdefghij1234567890XYZ"
    )
    kinds = {f.kind for f in secret_scan.scan(text)}
    assert {"anthropic-key", "openai-key", "tavily-key", "email", "bearer-token"} <= kinds


def test_anthropic_key_not_double_counted_as_openai():
    """An sk-ant- key must register once (anthropic), not also as an openai sk- key."""
    finds = secret_scan.scan("key sk-ant-abcdefghij1234567890ABCD here")
    kinds = [f.kind for f in finds]
    assert kinds.count("anthropic-key") == 1
    assert "openai-key" not in kinds


def test_preview_never_exposes_full_secret():
    secret = "sk-ant-abcdefghij1234567890ABCDEFGH"
    f = secret_scan.scan(f"x {secret} y")[0]
    assert secret not in f.preview
    assert "…" in f.preview


def test_no_false_positive_on_ordinary_prose():
    assert secret_scan.scan("The quick brown fox jumps over 12 lazy dogs.") == []


# ── textutil.mask_secret — THE one masking rule ───────────────────────────────────────────────


def test_mask_secret_one_envelope():
    from textutil import mask_secret

    assert mask_secret("short") == "****"  # ≤8 chars: show nothing at all
    long = "sk-abcdefghijklmnop"
    m = mask_secret(long)
    assert long not in m and "…" in m
    assert m.startswith(long[:4]) and m.endswith(long[-2:])
    assert mask_secret("") == "" and mask_secret(None) == ""


def test_mask_surface_delegates_to_it():
    """trust/secret_scan's findings render THE one masking rule."""
    from textutil import mask_secret
    from trust import secret_scan

    secret = "tvly-0123456789abcdef"
    assert secret_scan._mask(secret) == mask_secret(secret)
    assert secret_scan._mask("tiny") == "****"  # short secrets show nothing
