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
        from config import MODEL_ROLES

        before = {r: cfg.get(f"tiers.4b.roles.{r}") for r in MODEL_ROLES}
        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr(
            "commands.runtime._persist_bindings",
            lambda *a, **k: pytest.fail("a refused bind must not persist"),
        )
        runtime._bind(cfg, "all", "qwen3-coder:30b")

        # Not just "persist was not called": every role binding must be untouched in memory too,
        # or the session runs something the file never said (an assertion-free test used to pass
        # against a _bind that simply returned).
        assert {r: cfg.get(f"tiers.4b.roles.{r}") for r in MODEL_ROLES} == before

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


class TestTierMetrics:
    def test_tier_rows_carry_params_context_and_calibration(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows(cfg=_template_config())
        keys = [r["key"] for r in rows]
        assert keys == ["800m", "2b", "4b", "9b", "27b", "35b"]
        for row in rows:
            assert row["model"]
            assert row["params"]              # e.g. "27.3B"
            assert row["ctx"] == 32768
            assert row["max_ctx"] == 262144
            assert isinstance(row["calibrated"], bool)
            assert row["legacy"] is False

    def test_the_active_tier_is_marked(self):
        from commands.runtime import _tier_rows

        rows = _tier_rows(active="9b", cfg=_template_config())
        assert [r["key"] for r in rows if r["active"]] == ["9b"]


class TestLegacyTierAdviceIsActionable:
    """The migration warning's remediation must work for the population that SEES it — an
    upgrading user whose config.yaml still has laptop/workstation. The listing used to render
    six ladder rows none of which /models tier would accept, and the warning pointed at exactly
    that command."""

    @pytest.fixture(autouse=True)
    def _clean_migrations(self):
        import config

        config.clear_migrations()
        yield
        config.clear_migrations()

    def test_the_listing_shows_the_tiers_this_config_can_select(self):
        from commands.runtime import _tier_rows

        cfg = _legacy_config()
        rows = _tier_rows(cfg=cfg)
        assert [r["key"] for r in rows] == ["laptop", "workstation"]
        assert all(r["legacy"] for r in rows)
        # Every listed row is a key /models tier validates against — no unselectable rows.
        for row in rows:
            assert row["key"] in cfg.get("tiers", {})
        assert rows[1]["declared"] == "gemma4:31b"   # what the file says
        assert rows[1]["model"] == "qwen3.8:27b"     # what selecting it would actually run
        assert rows[1]["ctx"] == 32768               # …and that model's real window, not 8192

    def test_the_warning_points_at_a_command_that_works_here(self, monkeypatch):
        import config as config_mod
        from core import llms

        cfg = _legacy_config()
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
        cfg.model_for_role("synthesizer")

        problem = llms._migration_problems()[0]
        assert "/models tier" not in problem       # would answer "unknown tier" on this config
        assert "/models all qwen3.8:27b" in problem

    def test_the_advised_command_actually_rebinds_and_clears_the_warning(self, monkeypatch,
                                                                        printed):
        import config as config_mod
        from commands import runtime
        from config import MODEL_ROLES
        from core import llms

        cfg = _legacy_config()
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        for role in MODEL_ROLES:
            cfg.model_for_role(role)
        assert llms._migration_problems()          # the warning is live

        runtime._bind(cfg, "all", "qwen3.8:27b")   # exactly what the warning advises

        for role in MODEL_ROLES:
            assert cfg.get(f"tiers.workstation.roles.{role}") == "qwen3.8:27b"
            cfg.model_for_role(role)
        assert llms._migration_problems() == []    # and the warning is gone

    def test_a_ladder_config_still_gets_the_tier_advice(self, monkeypatch):
        import config as config_mod
        from core import llms

        monkeypatch.setattr(config_mod, "_config", _template_config(), raising=False)
        assert "/models tier" in llms._rebind_hint("qwen3.5:4b")


class TestConfigDoorIsGatedToo:
    """/config writes the very same `tiers.*.roles.*` keys /models refuses — and, unlike a trust
    key, persists by default. It used to write a binding the product refuses straight into
    config.yaml and read it back for the session."""

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

    def test_a_non_family_role_binding_is_refused_and_not_persisted(self, cfg, wired,
                                                                    monkeypatch):
        monkeypatch.setattr("commands.config._persist_key",
                            lambda *a, **k: pytest.fail("a refused bind must not persist"))
        self._run(["tiers.4b.roles.synthesizer", "gemma4:e4b"])

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:4b"   # unchanged in memory too
        blob = "\n".join(wired)
        assert "outside the supported model family" in blob
        assert "qwen3.5:4b" in blob                                    # the ladder as the fix

    def test_a_family_role_binding_still_sets_and_persists(self, cfg, wired, monkeypatch):
        saved = []
        monkeypatch.setattr("commands.config._persist_key",
                            lambda _cfg, key: saved.append(key))
        self._run(["tiers.4b.roles.synthesizer", "qwen3.5:9b"])

        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.5:9b"
        assert saved == ["tiers.4b.roles.synthesizer"]

    def test_the_embedder_key_stays_exempt(self, cfg, wired, monkeypatch):
        # Not a chat model: no raw-mode template, no logprobs, no calibration claim rides on it.
        monkeypatch.setattr("commands.config._persist_key", lambda *a, **k: None)
        self._run(["tiers.4b.embedder", "nomic-embed-text:v2"])

        assert cfg.get("tiers.4b.embedder") == "nomic-embed-text:v2"


class TestListingMetrics:
    """`tui/ui/readouts.show_models` grew a `meta` hook — and until 2026-08-16 nothing passed
    it, so /models list showed none of the metrics the spec asks for and the detail column had
    been widened 14 -> 26 for data that never arrived."""

    class _Local:
        def __init__(self, name, is_embedding=False):
            self.name = name
            self.is_embedding = is_embedding
            self.size_bytes = 1
            self.parameter_size = "4.7B"
            self.quantization = "Q4_K_M"
            self.family = "qwen"
            self.size_h = "3.4G"

    def test_meta_carries_windows_and_calibration_per_tag(self, isolated_paths):
        # isolated_paths: the calibration verdict also consults the USER overlay under
        # database/, and a real one on the developer's machine must not decide a test.
        from commands.runtime import _model_meta

        cfg = _template_config()
        meta = _model_meta([self._Local("qwen3.5:4b"), self._Local("mystery:7b")], cfg)

        assert meta["qwen3.5:4b"] == {"ctx": 32768, "max_ctx": 262144, "calibrated": True}
        assert meta["mystery:7b"]["calibrated"] is False

    def test_an_embedder_gets_no_calibration_verdict(self, isolated_paths):
        # Not a chat model: it produces no logprobs, so "uncalibrated" would read as a defect.
        from commands.runtime import _model_meta

        meta = _model_meta([self._Local("qwen3-embedding:8b", is_embedding=True)],
                           _template_config())
        assert "calibrated" not in meta["qwen3-embedding:8b"]

    def test_the_listing_actually_passes_it(self, cfg, monkeypatch):
        from commands import runtime

        seen = {}
        monkeypatch.setattr("core.llms.list_local_models",
                            lambda: [self._Local("qwen3.5:4b")])
        monkeypatch.setattr("core.llms.model_id", lambda role: "qwen3.5:4b")
        monkeypatch.setattr("tui.ui.show_models",
                            lambda *a, **k: seen.update(k))
        import config as config_mod

        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
        runtime._models(None, ["list"])

        assert seen["meta"]["qwen3.5:4b"]["ctx"]
        assert "calibrated" in seen["meta"]["qwen3.5:4b"]

    def test_the_renderer_shows_them_without_crashing(self, capsys):
        from tui import ui

        ui.show_models(
            [self._Local("qwen3.5:4b")], {"synthesizer": "qwen3.5:4b"}, "4b",
            "qwen3-embedding:8b",
            meta={"qwen3.5:4b": {"ctx": 32768, "max_ctx": 262144, "calibrated": True}},
        )
        out = capsys.readouterr().out
        assert "32k/256k" in out          # compact, so the bindings tail still fits
        assert "calibrated" in out
