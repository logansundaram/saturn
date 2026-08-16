"""The supported-family gate: the predicate, the size ladder, and the migration map."""

import pytest

from core import model_family as mf


class TestInFamily:
    def test_every_ladder_tag_is_in_family(self):
        for _key, tag in mf.SIZE_LADDER:
            assert mf.in_family(tag), tag

    def test_bare_family_name_matches(self):
        assert mf.in_family("qwen3.6")

    def test_case_insensitive(self):
        assert mf.in_family("QWEN3.5:0.8B")

    def test_matching_is_anchored_not_a_loose_prefix(self):
        # qwen3.50 must NOT satisfy a qwen3.5 test — the whole point of anchoring.
        assert not mf.in_family("qwen3.50:1b")

    @pytest.mark.parametrize(
        "tag",
        ["gemma4:e4b", "qwen3-coder:30b", "qwen3-embedding:8b", "qwen3.7:9b", "", "   "],
    )
    def test_outsiders_rejected(self, tag):
        assert not mf.in_family(tag)

    def test_none_is_not_in_family(self):
        assert not mf.in_family(None)


class TestLadder:
    def test_classes_match_the_ladder_order(self):
        assert mf.classes() == ("800m", "2b", "4b", "9b", "27b", "35b")

    def test_tag_for_round_trips(self):
        for key, tag in mf.SIZE_LADDER:
            assert mf.tag_for(key) == tag

    def test_tag_for_is_case_insensitive(self):
        assert mf.tag_for("800M") == "qwen3.5:0.8B"

    def test_tag_for_preserves_the_capital_b_tag(self):
        # Ollama tags are case-sensitive: the 0.8B tag must survive verbatim.
        assert mf.tag_for("800m") == "qwen3.5:0.8B"

    def test_tag_for_unknown_class_raises(self):
        with pytest.raises(KeyError):
            mf.tag_for("13b")

    def test_default_class_is_on_the_ladder(self):
        assert mf.DEFAULT_CLASS in mf.classes()

    def test_no_size_class_key_contains_a_dot(self):
        # config.get/set/persist parse dotted paths, so a "." in a tier key splits it into two
        # segments and role binds write to the wrong place. Keep class keys dot-free.
        for key in mf.classes():
            assert "." not in key, key


class TestMigrate:
    @pytest.mark.parametrize(
        "old,expected",
        [
            ("gemma4:e2b", "2b"),
            ("gemma4:e4b", "4b"),
            ("gemma4:12b", "9b"),
            ("gemma4:26b", "27b"),
            ("gemma4:31b", "27b"),
            ("qwen3-coder:30b", "27b"),
        ],
    )
    def test_legacy_table_is_exact(self, old, expected):
        assert mf.migrate(old) == expected

    def test_legacy_lookup_is_case_insensitive(self):
        assert mf.migrate("GEMMA4:E4B") == "4b"

    def test_unknown_tag_falls_back_to_the_size_parse(self):
        # |33 - 27.3| = 5.7 vs |33 - 36.0| = 3.0 -> nearest class is 35b, not 27b.
        assert mf.migrate("mystery:33b") == "35b"
        assert mf.migrate("mystery:3b") == "2b"

    def test_unparseable_tag_falls_back_to_the_default_class(self):
        assert mf.migrate("devstral-small-2:latest") == mf.DEFAULT_CLASS

    def test_migrate_always_returns_a_real_class(self):
        for tag in ["gemma4:e4b", "mystery:33b", "junk", "", None]:
            assert mf.migrate(tag) in mf.classes()


class TestConfigMigrationSeam:
    """config.model_for_role is THE migration seam: substitute in memory, record it, never
    rewrite config.yaml."""

    def _cfg(self, synth="qwen3.8:27b"):
        from config import Config

        return Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "planner": synth, "tool_caller": synth, "synthesizer": synth,
                "utility": synth, "judge": synth,
            }, "embedder": "qwen3-embedding:8b"}},
            "capabilities": {},
        })

    def setup_method(self):
        import config

        config.clear_migrations()

    def test_family_binding_passes_through_untouched(self):
        import config

        spec = self._cfg().model_for_role("synthesizer")
        assert spec.model == "qwen3.8:27b"
        assert config.migrated_bindings() == {}

    def test_non_family_binding_is_substituted(self):
        spec = self._cfg("gemma4:e4b").model_for_role("synthesizer")
        assert spec.model == "qwen3.5:4b"
        assert spec.provider == "ollama"

    def test_the_substitution_is_recorded(self):
        import config

        self._cfg("qwen3-coder:30b").model_for_role("synthesizer")
        assert config.migrated_bindings() == {"qwen3-coder:30b": "qwen3.8:27b"}

    def test_a_non_ollama_binding_is_left_for_the_cloud_shelve_refusal(self):
        import config
        from config import Config

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "synthesizer": {"provider": "anthropic", "model": "claude-sonnet-4"},
            }}},
        })
        spec = cfg.model_for_role("synthesizer")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-sonnet-4"
        assert config.migrated_bindings() == {}

    def test_the_embedder_is_exempt(self):
        cfg = self._cfg()
        assert cfg.embedder_model == "qwen3-embedding:8b"

    def test_capability_max_context_window_defaults_to_the_runtime_window(self):
        from config import Config

        cfg = Config({"capabilities": {"m": {"context_window": 32768}}})
        cap = cfg.capability_of("m")
        assert cap.context_window == 32768
        assert cap.max_context_window == 32768

    def test_capability_max_context_window_is_read_when_present(self):
        from config import Config

        cfg = Config({"capabilities": {"m": {"context_window": 32768,
                                             "max_context_window": 262144}}})
        cap = cfg.capability_of("m")
        assert cap.context_window == 32768        # what num_ctx_for returns — unchanged
        assert cap.max_context_window == 262144   # display only

    def test_num_ctx_for_still_returns_the_runtime_window_not_the_max(self):
        from config import Config

        cfg = Config({"runtime": {"num_ctx": None},
                      "capabilities": {"m": {"context_window": 32768,
                                             "max_context_window": 262144}}})
        assert cfg.num_ctx_for("m") == 32768


class TestShippedConfigMatchesTheLadder:
    """The template config and the ladder must not drift apart — a tier binding a tag with no
    capabilities entry silently runs at the conservative 8192 default."""

    def _template(self):
        import pathlib

        import yaml

        root = pathlib.Path(__file__).resolve().parents[1]
        return yaml.safe_load((root / "config.default.yaml").read_text(encoding="utf-8"))

    def test_tier_keys_are_exactly_the_size_classes(self):
        from core import model_family as mf

        assert tuple(self._template()["tiers"]) == mf.classes()

    def test_every_tier_binds_its_ladder_tag_on_every_role(self):
        from config import MODEL_ROLES
        from core import model_family as mf

        tiers = self._template()["tiers"]
        for key, tag in mf.SIZE_LADDER:
            roles = tiers[key]["roles"]
            assert set(roles) == set(MODEL_ROLES), key
            assert set(roles.values()) == {tag}, key

    def test_every_role_binding_is_in_family(self):
        from core import model_family as mf

        for key, tier in self._template()["tiers"].items():
            for role, tag in tier["roles"].items():
                assert mf.in_family(tag), f"{key}.{role} = {tag}"

    def test_every_ladder_tag_has_a_capabilities_entry(self):
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert tag in caps, tag

    def test_capabilities_keep_the_runtime_window_off_the_architectural_max(self):
        # Collapsing these is a latent OOM: 262144 num_ctx exhausts consumer VRAM.
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert caps[tag]["context_window"] == 32768, tag
            assert caps[tag]["max_context_window"] == 262144, tag

    def test_retired_models_are_gone_from_the_template(self):
        template = self._template()
        text = str(template)
        for retired in ("gemma4", "qwen3-coder", "bench-coder"):
            assert retired not in text, retired

    def test_the_default_tier_is_the_default_class(self):
        from core import model_family as mf

        assert self._template()["active_tier"] == mf.DEFAULT_CLASS

    def test_the_embedder_is_unchanged_on_every_tier(self):
        for key, tier in self._template()["tiers"].items():
            assert tier["embedder"] == "qwen3-embedding:8b", key


class TestStartupReportsMigrations:
    def setup_method(self):
        import config

        config.clear_migrations()

    def teardown_method(self):
        import config

        config.clear_migrations()

    def test_no_migration_reports_nothing(self):
        from core import llms

        assert llms._migration_problems() == []

    def test_a_migration_is_reported_with_both_ids_and_the_fix(self):
        import config
        from config import Config
        from core import llms

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {"synthesizer": "gemma4:e4b"}}},
        })
        cfg.model_for_role("synthesizer")

        problems = llms._migration_problems()
        assert len(problems) == 1
        line = problems[0]
        assert "gemma4:e4b" in line          # what the file says
        assert "qwen3.5:4b" in line          # what is actually running
        assert "/models tier" in line        # how to make it permanent
        assert "config.yaml" in line         # and that the file was NOT rewritten

    def test_each_distinct_substitution_is_reported_once(self):
        from config import Config
        from core import llms

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "planner": "gemma4:e4b", "synthesizer": "gemma4:e4b",
                "judge": "qwen3-coder:30b",
            }}},
        })
        for role in ("planner", "synthesizer", "judge"):
            cfg.model_for_role(role)

        assert len(llms._migration_problems()) == 2


class TestTemplateRegistryAgreesWithTheFamily:
    """The freeze hotkey must arm for every bindable model and no others — two lists that drift
    would either strand a supported model or promise continuation for an unbindable one."""

    def test_template_prefixes_are_exactly_the_family(self):
        from core import chat_template, model_family as mf

        covered = tuple(p for t in chat_template.TEMPLATES for p in t.prefixes)
        assert sorted(covered) == sorted(mf.FAMILY_PREFIXES)

    def test_every_ladder_tag_is_supported_for_continuation(self):
        from core import chat_template, model_family as mf

        for _key, tag in mf.SIZE_LADDER:
            assert chat_template.supported(tag), tag

    def test_a_retired_family_is_no_longer_supported(self):
        from core import chat_template

        assert not chat_template.supported("gemma4:e4b")
