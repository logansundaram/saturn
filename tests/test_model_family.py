"""The size ladder: the class table, off-ladder pricing, and the shipped template."""

import pytest

from core import model_family as mf


class TestLadder:
    def test_classes_match_the_ladder_order(self):
        assert mf.classes() == ("4b", "9b", "27b", "35b")

    def test_tag_for_round_trips(self):
        for key, tag in mf.SIZE_LADDER:
            assert mf.tag_for(key) == tag

    def test_tag_for_is_case_insensitive(self):
        assert mf.tag_for("27B") == "qwen3.8:27b"

    def test_tag_for_unknown_class_raises(self):
        with pytest.raises(KeyError):
            mf.tag_for("13b")

    def test_default_class_is_on_the_ladder(self):
        assert mf.DEFAULT_CLASS in mf.classes()

    def test_the_parameter_table_covers_exactly_the_ladder(self):
        # A stray or missing key makes class_of() price an off-ladder tag against a class that
        # does not exist.
        assert set(mf._CLASS_PARAMS) == set(mf.classes())

    def test_no_size_class_key_contains_a_dot(self):
        # config.get/set/persist parse dotted paths, so a "." in a tier key splits it into two
        # segments and role binds write to the wrong place. Keep class keys dot-free.
        for key in mf.classes():
            assert "." not in key, key


class TestClassOf:
    def test_a_ladder_tag_is_its_own_class(self):
        for key, tag in mf.SIZE_LADDER:
            assert mf.class_of(tag) == key, tag

    def test_an_off_ladder_tag_prices_as_the_nearest_size(self):
        # |33 - 27.3| = 5.7 vs |33 - 36.0| = 3.0 -> nearest class is 35b, not 27b.
        assert mf.class_of("mystery:33b") == "35b"
        assert mf.class_of("mystery:3b") == "4b"
        assert mf.class_of("gemma4:e4b") == "4b"
        assert mf.class_of("GEMMA4:31B") == "27b"

    def test_unparseable_tag_falls_back_to_the_default_class(self):
        assert mf.class_of("devstral-small-2:latest") == mf.DEFAULT_CLASS

    def test_class_of_always_returns_a_real_class(self):
        for tag in ["gemma4:e4b", "mystery:33b", "junk", "", None]:
            assert mf.class_of(tag) in mf.classes()


class TestConfigResolution:
    """config.model_for_role hands back what config.yaml binds — on or off the ladder — and
    never rewrites the file."""

    def _cfg(self, synth="qwen3.8:27b"):
        from config import Config

        return Config({
            "active_tier": "t",
            "tiers": {"t": {"model": synth, "embedder": "qwen3-embedding:8b"}},
            "capabilities": {},
        })

    def test_a_ladder_binding_passes_through_untouched(self):
        assert self._cfg().chat_model == "qwen3.8:27b"

    def test_an_off_ladder_binding_passes_through_untouched(self):
        assert self._cfg("gemma4:e4b").chat_model == "gemma4:e4b"

    def test_a_provider_mapping_from_a_pre_cut_config_refuses_actionably(self):
        import pytest
        from config import Config

        cfg = Config({"active_tier": "t", "tiers": {"t": {
            "model": {"provider": "anthropic", "model": "claude-sonnet-4"}}}})
        with pytest.raises(KeyError, match="bare Ollama model id"):
            cfg.chat_model

    def test_an_old_roles_block_is_refused_with_the_line_to_write(self):
        """The pre-2026-09-30 `roles:` shape is no longer read: the error names the exact
        replacement line, built from the agent's old entry."""
        import pytest
        from config import Config

        cfg = Config({"active_tier": "t", "tiers": {"t": {"roles": {
            "planner": "old:1b", "tool_caller": "qwen3.5:9b", "utility": "qwen3.5:4b"}}}})
        with pytest.raises(KeyError, match='old `roles:` block .* model: "qwen3.5:9b"'):
            cfg.chat_model

    def test_a_tier_without_a_model_refuses_actionably(self):
        import pytest
        from config import Config

        with pytest.raises(KeyError, match="defines no model"):
            Config({"active_tier": "t", "tiers": {"t": {"embedder": "e"}}}).chat_model

    def test_the_embedder_is_exempt(self):
        cfg = self._cfg()
        assert cfg.embedder_model == "qwen3-embedding:8b"

    def test_a_config_without_an_active_tier_falls_back_to_the_default_class(self):
        """A config missing the key must resolve to a tier that exists, not hard-fail on every
        model resolution."""
        from config import Config
        from core import model_family as mf

        cfg = Config({"tiers": {key: {"model": tag, "embedder": "qwen3-embedding:8b"}
                                for key, tag in mf.SIZE_LADDER}})
        assert cfg.active_tier == mf.DEFAULT_CLASS
        assert cfg.chat_model == mf.tag_for(mf.DEFAULT_CLASS)

    def test_num_ctx_for_returns_the_runtime_window_not_the_max(self):
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

    def test_every_tier_binds_its_ladder_tag(self):
        from core import model_family as mf

        tiers = self._template()["tiers"]
        for key, tag in mf.SIZE_LADDER:
            assert tiers[key]["model"] == tag, key
            assert "roles" not in tiers[key], key

    def test_every_ladder_tag_has_a_capabilities_entry(self):
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert tag in caps, tag

    def test_capabilities_keep_the_runtime_window_off_the_architectural_max(self):
        # A 262144 num_ctx (every tag's architectural max) exhausts consumer VRAM: the runtime
        # windows step up the ladder but every one stays far below that ceiling.
        from core import model_family as mf

        caps = self._template()["capabilities"]
        expected = {"4b": 32768,
                    "9b": 65536, "27b": 65536, "35b": 131072}
        for key, tag in mf.SIZE_LADDER:
            assert caps[tag]["context_window"] == expected[key], tag

    def test_runtime_windows_fit_each_tier_on_its_home_hardware(self):
        """The window is a memory decision: at the template's window every class must fit the
        machine it is meant for (core/hardware.py prices it), so a window bump here can't
        silently push a tier off its hardware."""
        from core import hardware, model_family as mf

        caps = self._template()["capabilities"]
        windows = {key: caps[tag]["context_window"] for key, tag in mf.SIZE_LADDER}
        home = {"4b": 6.0, "9b": 12.0, "27b": 22.5, "35b": 27.0}
        for key, budget in home.items():
            assert hardware.need_gb(key, windows[key]) <= budget, key

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
