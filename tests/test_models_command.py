"""/models — the family refusal and the metrics readout."""

import pytest


@pytest.fixture
def printed(monkeypatch):
    """Capture the command module's output lines."""
    lines = []
    monkeypatch.setattr("commands.runtime._print", lambda line="": lines.append(str(line)))
    return lines


@pytest.fixture
def cfg():
    from config import Config

    return Config({
        "active_tier": "4b",
        "tiers": {"4b": {"provider": "ollama", "roles": {
            "planner": "qwen3.5:4b", "tool_caller": "qwen3.5:4b",
            "synthesizer": "qwen3.5:4b", "utility": "qwen3.5:4b", "judge": "qwen3.5:4b",
        }, "embedder": "qwen3-embedding:8b"}},
        "capabilities": {},
    })


class TestBindRefusal:
    def test_a_non_family_bind_is_refused(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        runtime._bind(cfg, "synthesizer", "gemma4:e4b")

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:4b"   # unchanged
        blob = "\n".join(printed)
        assert "gemma4:e4b" in blob
        assert "qwen3.5:4b" in blob      # the ladder is shown as the fix

    def test_the_refusal_never_persists_anything(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr(
            "commands.runtime._persist_bindings",
            lambda *a, **k: pytest.fail("a refused bind must not persist"),
        )
        runtime._bind(cfg, "all", "qwen3-coder:30b")

    def test_a_family_bind_still_works(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "synthesizer", "qwen3.5:9b")

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:9b"

    def test_the_embedder_is_exempt_from_the_family_gate(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "embedder", "qwen3-embedding:4b")

        assert cfg.get("tiers.4b.embedder") == "qwen3-embedding:4b"


class TestTierMetrics:
    def test_tier_rows_carry_params_context_and_calibration(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows()
        keys = [r["key"] for r in rows]
        assert keys == ["800m", "2b", "4b", "9b", "27b", "35b"]
        for row in rows:
            assert row["model"]
            assert row["params"]              # e.g. "27.3B"
            assert row["ctx"] == 32768
            assert row["max_ctx"] == 262144
            assert isinstance(row["calibrated"], bool)

    def test_the_active_tier_is_marked(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows(active="9b")
        assert [r["key"] for r in rows if r["active"]] == ["9b"]
