"""The size ladder: the class table, off-ladder pricing, and the shipped template."""

import pytest

from core import model_family as mf


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

    def test_the_parameter_table_covers_exactly_the_ladder(self):
        # A stray or missing key makes class_of() price an off-ladder tag against a class that
        # does not exist.
        assert set(mf._CLASS_PARAMS) == set(mf.classes())

    def test_is_ladder_tag_matches_the_ladder_case_insensitively(self):
        for _key, tag in mf.SIZE_LADDER:
            assert mf.is_ladder_tag(tag)
            assert mf.is_ladder_tag(tag.upper())
        assert not mf.is_ladder_tag("qwen3.5:99b")  # a tag we do not ship is not a ladder tag
        assert not mf.is_ladder_tag("")

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
        assert mf.class_of("mystery:3b") == "2b"
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
            "tiers": {"t": {"provider": "ollama", "roles": {
                "tool_caller": synth, "utility": synth,
            }, "embedder": "qwen3-embedding:8b"}},
            "capabilities": {},
        })

    def test_a_ladder_binding_passes_through_untouched(self):
        spec = self._cfg().model_for_role("tool_caller")
        assert spec.model == "qwen3.8:27b"

    def test_an_off_ladder_binding_passes_through_untouched(self):
        spec = self._cfg("gemma4:e4b").model_for_role("tool_caller")
        assert spec.model == "gemma4:e4b"
        assert spec.provider == "ollama"

    def test_a_non_ollama_binding_is_left_for_the_cloud_shelve_refusal(self):
        from config import Config

        cfg = Config({
            "active_tier": "t",
            "tiers": {"t": {"provider": "ollama", "roles": {
                "tool_caller": {"provider": "anthropic", "model": "claude-sonnet-4"},
            }}},
        })
        spec = cfg.model_for_role("tool_caller")
        assert spec.provider == "anthropic"
        assert spec.model == "claude-sonnet-4"

    def test_the_embedder_is_exempt(self):
        cfg = self._cfg()
        assert cfg.embedder_model == "qwen3-embedding:8b"

    def test_a_config_without_an_active_tier_falls_back_to_the_default_class(self):
        """The fallback used to be "workstation", a preset that stopped shipping with the size
        ladder — so a config missing the key named a tier that does not exist and hard-failed on
        every model resolution."""
        from config import Config, MODEL_ROLES
        from core import model_family as mf

        cfg = Config({"tiers": {key: {"provider": "ollama",
                                      "roles": {r: tag for r in MODEL_ROLES},
                                      "embedder": "qwen3-embedding:8b"}
                                for key, tag in mf.SIZE_LADDER}})
        assert cfg.active_tier == mf.DEFAULT_CLASS
        assert cfg.model_for_role("tool_caller").model == mf.tag_for(mf.DEFAULT_CLASS)

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

    def test_every_ladder_tag_has_a_capabilities_entry(self):
        from core import model_family as mf

        caps = self._template()["capabilities"]
        for _key, tag in mf.SIZE_LADDER:
            assert tag in caps, tag

    def test_the_family_capability_fallback_matches_the_template(self):
        """A config predating the family lock has no capabilities entry for the tag a legacy
        binding is SUBSTITUTED with; the generic default would quarter its window to 8192,
        undisclosed. The fallback constants must therefore stay equal to what ships."""
        import config

        caps = self._template()["capabilities"]
        # The runtime windows step up the ladder (2026-09-01); the fallback is the SMALLEST
        # shipped window, so a tag with no entry is never handed more cache than any tier ships.
        shipped = [caps[tag]["context_window"] for _key, tag in self._ladder()]
        assert config.FAMILY_CONTEXT_WINDOW == min(shipped)
        for _key, tag in self._ladder():
            assert caps[tag]["max_context_window"] == config.FAMILY_MAX_CONTEXT_WINDOW, tag

    def test_a_ladder_tag_with_no_capabilities_entry_gets_the_family_defaults(self):
        import config
        from config import Config
        from core import model_family as mf

        cfg = Config({"capabilities": {}})              # an upgrader's config.yaml
        cap = cfg.capability_of(mf.tag_for(mf.DEFAULT_CLASS))
        assert cap.context_window == config.FAMILY_CONTEXT_WINDOW
        assert cap.max_context_window == config.FAMILY_MAX_CONTEXT_WINDOW
        # Everything else keeps the conservative default — this is a family fallback, not a
        # blanket one.
        assert cfg.capability_of("something-else:7b").context_window == 8192

    def _ladder(self):
        from core import model_family as mf

        return mf.SIZE_LADDER

    def test_capabilities_keep_the_runtime_window_off_the_architectural_max(self):
        # Collapsing these is a latent OOM: 262144 num_ctx exhausts consumer VRAM. The runtime
        # windows step up the ladder (2026-09-01) but every one stays far below the ceiling.
        from core import model_family as mf

        caps = self._template()["capabilities"]
        expected = {"800m": 32768, "2b": 32768, "4b": 32768,
                    "9b": 65536, "27b": 65536, "35b": 131072}
        for key, tag in mf.SIZE_LADDER:
            assert caps[tag]["context_window"] == expected[key], tag
            assert caps[tag]["max_context_window"] == 262144, tag

    def test_runtime_windows_fit_each_tier_on_its_home_hardware(self):
        """The window is a memory decision: at the template's window every class must fit the
        machine it is meant for (core/hardware.py prices it), so a window bump here can't
        silently push a tier off its hardware."""
        from core import hardware, model_family as mf

        caps = self._template()["capabilities"]
        windows = {key: caps[tag]["context_window"] for key, tag in mf.SIZE_LADDER}
        home = {"800m": 3.0, "2b": 6.0, "4b": 6.0, "9b": 12.0, "27b": 22.5, "35b": 27.0}
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
