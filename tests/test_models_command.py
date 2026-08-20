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


class _Local:
    """A stand-in for `llms.LocalModel` (one stub, shared by every listing test)."""

    def __init__(self, name, is_embedding=False):
        self.name = name
        self.is_embedding = is_embedding
        self.size_bytes = 1
        self.parameter_size = "4.7B"
        self.quantization = "Q4_K_M"
        self.family = "qwen"
        self.size_h = "3.4G"


class TestListingMetrics:
    """`tui/ui/readouts.show_models` grew a `meta` hook — and until 2026-08-16 nothing passed
    it, so /models list showed none of the metrics the spec asks for and the detail column had
    been widened 14 -> 26 for data that never arrived."""

    _Local = _Local

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


class TestFamilyFilteredListing:
    """`/models` used to render `ollama list` verbatim — every pulled tag, including the ones
    `_bind` refuses at the family gate, so the picker's numbers pointed at a refusal. The filter
    is the LADDER (one tag per size class, the most recent), not the whole family: two tags of
    the same size are the same hardware cost and one is simply older."""

    _Local = _Local

    def _pulled(self):
        return [
            self._Local("qwen3.5:4b"),
            self._Local("qwen3.6:27b"),         # superseded at 27b by qwen3.8:27b
            self._Local("qwen3.6:35b"),
            self._Local("qwen3.8:27b"),
            self._Local("qwen3-embedding:8b", is_embedding=True),
            self._Local("gemma4:12b"),
            self._Local("gpt-oss:20b"),
            self._Local("qwen2.5:3b"),          # NOT the family: anchored prefix, not a substring
        ]

    def test_only_the_ladder_tags_and_embedders_are_offered(self):
        from commands.runtime import _selectable

        shown, hidden = _selectable(self._pulled(), {"synthesizer": "qwen3.5:4b"},
                                    "qwen3-embedding:8b")

        assert [m.name for m in shown] == [
            "qwen3.5:4b", "qwen3.6:35b", "qwen3.8:27b", "qwen3-embedding:8b",
        ]
        assert hidden == 4

    def test_one_row_per_size_class(self):
        # The point of the ladder filter: 27b appears ONCE, as the most recent tag at that size.
        # (Every unbound chat row IS a ladder tag, and the ladder carries one tag per size by
        # construction — no `migrate` round-trip, which is documented for NON-family ids only.)
        from commands.runtime import _selectable
        from core import model_family

        shown, _ = _selectable(self._pulled(), {}, "")
        chat = [m.name for m in shown if not m.is_embedding]

        assert "qwen3.6:27b" not in chat
        assert "qwen3.8:27b" in chat
        assert set(chat) <= {tag for _key, tag in model_family.SIZE_LADDER}

    def test_a_superseded_tag_is_still_bindable_by_name(self, cfg, printed, monkeypatch):
        # Hidden from the table is not refused at the gate — and the bind must actually LAND
        # (a bare model_family assertion would stay green if _bind's gate were ever tightened
        # from in_family to the ladder, silently breaking the documented contract).
        from commands import runtime
        from core import model_family

        assert not model_family.is_ladder_tag("qwen3.6:27b")     # hidden from the table…
        monkeypatch.setattr("core.llms.reset_models", lambda: None)
        monkeypatch.setattr("commands.runtime._persist_bindings", lambda *a, **k: None)
        monkeypatch.setattr("commands.runtime._resync_rag_after_model_change", lambda: None)
        runtime._bind(cfg, "all", "qwen3.6:27b")                 # …but binds by name
        assert cfg.get("tiers.4b.roles.synthesizer") == "qwen3.6:27b"

    def test_a_bound_tag_is_never_hidden(self):
        # The `◂ all roles` tail must never sit on a row the filter dropped.
        from commands.runtime import _selectable

        shown, hidden = _selectable(self._pulled(), {"synthesizer": "gemma4:12b"}, "")

        assert "gemma4:12b" in [m.name for m in shown]
        assert hidden == 3

    def test_bound_matching_is_case_insensitive(self):
        # is_ladder_tag folds case; the bound/declared clauses must too — a casing drift between
        # config.yaml and the daemon's reported tag must not hide a bound row.
        from commands.runtime import _selectable

        shown, hidden = _selectable([self._Local("MyVectors:Latest")], {}, "myvectors:latest")

        assert [m.name for m in shown] == ["MyVectors:Latest"]
        assert hidden == 0

    def test_a_declared_legacy_binding_stays_visible(self, cfg, printed, monkeypatch):
        # PRODUCTION shape: config.yaml literally names gemma4:12b, but model_id reports the
        # FAMILY-SUBSTITUTED ladder tag (config._enforce_family), so only the DECLARED ids can
        # keep the pulled legacy tag visible — "a legacy binding has to be visible to be
        # understood" must hold on the real call path, not just on raw injected bindings.
        from commands import runtime
        import config as config_mod
        from config import MODEL_ROLES

        for role in MODEL_ROLES:
            cfg.set(f"tiers.4b.roles.{role}", "gemma4:12b")
        monkeypatch.setattr("core.llms.list_local_models", self._pulled)
        monkeypatch.setattr("core.llms.model_id", lambda role: "qwen3.5:9b")   # what RUNS
        rendered = {}
        monkeypatch.setattr("tui.ui.show_models",
                            lambda models, *a, **k: rendered.update(models=models))
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)

        runtime._models(None, ["list"])

        assert "gemma4:12b" in [m.name for m in rendered["models"]]

    def test_all_filtered_empty_state_does_not_misdiagnose(self, capsys):
        # Daemon up, every pulled model filtered out: the old "(no local models — is the Ollama
        # daemon running?)" hint would sit directly above a "(N other installed models hidden)"
        # note — a contradiction. With hidden > 0 the empty state must say "nothing offerable"
        # and point at the by-name binds instead.
        from tui import ui

        ui.show_models([], {"synthesizer": "qwen3.5:9b"}, "9b", "", hidden=2)
        out = capsys.readouterr().out

        assert "is the Ollama daemon running" not in out
        assert "bind one by name" in out

    def test_the_drop_is_disclosed_not_silent(self, cfg, printed, monkeypatch):
        from commands import runtime
        import config as config_mod

        monkeypatch.setattr("core.llms.list_local_models", self._pulled)
        monkeypatch.setattr("core.llms.model_id", lambda role: "qwen3.5:4b")
        rendered = {}
        monkeypatch.setattr("tui.ui.show_models",
                            lambda models, *a, **k: rendered.update(models=models))
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)

        runtime._models(None, ["list"])

        assert [m.name for m in rendered["models"]] == [
            "qwen3.5:4b", "qwen3.6:35b", "qwen3.8:27b", "qwen3-embedding:8b",
        ]
        assert any("4 other installed models hidden" in line for line in printed)

    def test_the_picker_indexes_the_filtered_list(self, cfg, printed, monkeypatch):
        # The number the user types must select the row they SAW, not a row of the raw inventory.
        from commands import runtime
        import config as config_mod

        monkeypatch.setattr("core.llms.list_local_models", self._pulled)
        monkeypatch.setattr("core.llms.model_id", lambda role: "qwen3.5:4b")
        monkeypatch.setattr("tui.ui.show_models", lambda *a, **k: None)
        monkeypatch.setattr(config_mod, "_config", cfg, raising=False)
        answers = iter(["3", "synthesizer"])   # the 3rd SHOWN row, not the 3rd pulled tag
        monkeypatch.setattr("tui.ui.ask", lambda *a, **k: next(answers))
        bound = {}
        monkeypatch.setattr(runtime, "_bind",
                            lambda cfg_, target, model, **k: bound.update(t=target, m=model))

        runtime._models(None, [])

        assert bound == {"t": "synthesizer", "m": "qwen3.8:27b"}
