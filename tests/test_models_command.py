"""/models — binding, the legacy-tier advice, and the /config door."""

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
            "tool_caller": "qwen3.5:4b", "utility": "qwen3.5:4b",
        }, "embedder": "qwen3-embedding:8b"}},
        "capabilities": {},
    })


class TestBind:
    def test_a_bind_sets_the_role_key(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "tool_caller", "qwen3.5:9b")

        assert cfg.get("tiers.4b.roles.tool_caller") == "qwen3.5:9b"

    def test_the_embedder_binds_machine_wide(self, cfg, printed, monkeypatch):
        from commands import runtime

        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "embedder", "qwen3-embedding:4b")

        assert cfg.get("tiers.4b.embedder") == "qwen3-embedding:4b"


def _template_config():
    """A Config over the TRACKED template, never the developer's untracked config.yaml — the
    metrics tests assert exact windows, which a local edit would otherwise decide."""
    import pathlib

    import yaml

    from config import Config

    root = pathlib.Path(__file__).resolve().parents[1]
    return Config(yaml.safe_load((root / "config.default.yaml").read_text(encoding="utf-8")))


def _legacy_config():
    """What an upgrading user's config.yaml looks like: tier names from before the size ladder,
    bound to models that no longer exist in the family."""
    from config import Config, MODEL_ROLES

    def tier(model):
        return {"provider": "ollama", "roles": {r: model for r in MODEL_ROLES},
                "embedder": "qwen3-embedding:8b"}

    return Config({
        "active_tier": "workstation",
        "tiers": {"laptop": tier("gemma4:e4b"), "workstation": tier("gemma4:31b")},
        "capabilities": {},
    })


class TestLegacyTierAdviceIsActionable:
    """An upgrading user whose config.yaml still has laptop/workstation: the listing renders the
    ladder rows, none of which /models tier would accept on that config, so the page must name
    the bind that does work there."""

    def test_the_page_names_the_legacy_tiers_and_the_bind_that_works(self, printed, monkeypatch):
        from commands import runtime
        from core.hardware import HardwareProfile

        cfg = _legacy_config()
        notes = []
        monkeypatch.setattr("tui.ui.section", lambda *a, **k: None)
        monkeypatch.setattr("tui.ui.table", lambda *a, **k: None)
        monkeypatch.setattr("tui.ui.note", lambda m: notes.append(m))
        monkeypatch.setattr("tui.ui.warn", lambda m: notes.append(m))
        monkeypatch.setattr(runtime, "_probe",
                            lambda: HardwareProfile("Darwin", "arm64", "Apple M2", 8, 16.0, "", None, "apple"))
        monkeypatch.setattr("core.llms.ollama_reachable", lambda: False)

        runtime._models_page(cfg, prompt=False)

        blob = "\n".join(notes)
        assert "laptop" in blob and "workstation" in blob
        assert "/models all <tag>" in blob                 # the bind that works on this config
        # …and what selecting the legacy tier would actually run, not what the file says
        assert runtime._tier_binding(cfg, "27b") == ("", "qwen3.8:27b")


class TestConfigDoorBinds:
    """/config writes the very same `tiers.*.roles.*` keys /models does — and, unlike a trust
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

    def test_a_role_binding_sets_and_persists(self, cfg, wired, monkeypatch):
        saved = []
        monkeypatch.setattr("commands.config._persist_key",
                            lambda _cfg, key: saved.append(key))
        self._run(["tiers.4b.roles.tool_caller", "qwen3.5:9b"])

        assert cfg.get("tiers.4b.roles.tool_caller") == "qwen3.5:9b"
        assert saved == ["tiers.4b.roles.tool_caller"]

    def test_the_embedder_key_binds_too(self, cfg, wired, monkeypatch):
        monkeypatch.setattr("commands.config._persist_key", lambda *a, **k: None)
        self._run(["tiers.4b.embedder", "nomic-embed-text:v2"])

        assert cfg.get("tiers.4b.embedder") == "nomic-embed-text:v2"
