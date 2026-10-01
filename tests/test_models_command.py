"""/models — binding, and the /config door."""

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
        "tiers": {"4b": {"model": "qwen3.5:4b", "embedder": "qwen3-embedding:8b"}},
        "capabilities": {},
    })


class TestBind:
    def test_a_bind_sets_the_model_key(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.config._persist_key", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "model", "qwen3.5:9b")

        assert cfg.get("tiers.4b.model") == "qwen3.5:9b"
        assert cfg.chat_model == "qwen3.5:9b"

    def test_the_embedder_binds_machine_wide(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.config._persist_key", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "embedder", "qwen3-embedding:4b")

        assert cfg.get("tiers.4b.embedder") == "qwen3-embedding:4b"


class TestConfigDoorBinds:
    """/config writes the very same `tiers.*.model` key /models does — and, unlike a trust
    key, persists by default."""

    @pytest.fixture
    def wired(self, cfg, monkeypatch):
        import config as config_mod

        lines = []
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
        monkeypatch.setattr("commands.config._print", lambda line="": lines.append(str(line)))
        monkeypatch.setattr("commands.runtime._print", lambda line="": lines.append(str(line)))
        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.config._resync_rag_after_model_change", lambda: None)
        return lines

    def _run(self, args):
        from commands.config import _config

        _config(None, args)

    def test_a_model_binding_sets_and_persists(self, cfg, wired, monkeypatch):
        saved = []
        monkeypatch.setattr("commands.config._persist_key",
                            lambda _cfg, key: saved.append(key))
        self._run(["tiers.4b.model", "qwen3.5:9b"])

        assert cfg.chat_model == "qwen3.5:9b"
        assert saved == ["tiers.4b.model"]

    def test_the_embedder_key_binds_too(self, cfg, wired, monkeypatch):
        monkeypatch.setattr("commands.config._persist_key", lambda *a, **k: None)
        self._run(["tiers.4b.embedder", "nomic-embed-text:v2"])

        assert cfg.get("tiers.4b.embedder") == "nomic-embed-text:v2"
