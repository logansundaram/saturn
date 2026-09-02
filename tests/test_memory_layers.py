"""The layered memory store (stores/memory_registry, 2026-09-02) and its grounding contract.

PLAN.md's "prove it" list: grounding injects memory; `recall` filters; a fact survives a new
session (the store is the file, re-parsed cold); a superseding fact wins. Plus the loading rules
the design hangs on — user + open commitments + the memo digest always, agent/entities/negative
only by match, one cap with a trailer naming what didn't load, sensitive facts withheld when
inference is remote — and the per-fact metadata (provenance run, last-used, confirmed-count).
All offline; isolated_paths keeps the real memory.md untouched.
"""

from datetime import date, timedelta

import pytest
from langchain.messages import HumanMessage

from stores import memory_registry as mr


@pytest.fixture
def mem(isolated_paths):
    return isolated_paths


# ── layers, ids, supersession ───────────────────────────────────────────────────────────────


def test_add_assigns_stable_ids_and_layers(mem):
    assert mr.add_memory("I prefer terse answers").startswith("Remembered #1 (user)")
    assert mr.add_memory("Petra is my manager", layer="entities").startswith("Remembered #2 (entities)")
    mr.remove_memory(1)
    # ids are never reused — provenance and replaces= references stay meaningful.
    assert mr.add_memory("uses metric units").startswith("Remembered #3 (user)")
    assert [e["id"] for e in mr.entries()] == [2, 3]


def test_layer_spellings_fold_unknown_is_its_own_by_match_section_garble_is_user(mem):
    mr.add_memory("send the deck Friday", layer="todo")
    mr.add_memory("never use sudo", layer="don't")
    mr.add_memory("garbled", layer="???")
    mr.add_memory("pip needs --break-system-packages here", layer="Tools")
    layers = {e["text"]: e["layer"] for e in mr.entries()}
    assert layers == {"send the deck Friday": "commitments", "never use sudo": "negative",
                      "garbled": "user", "pip needs --break-system-packages here": "tools"}
    # A typo'd layer must never promote a fact into the always-loaded set: it loads by match.
    assert "pip needs" not in mr.read_memory_block("what time is it")
    assert "[tools]" in mr.read_memory_block("install it with pip")
    assert "## tools" in mr._read_raw()  # and it round-trips as its own section


def test_superseding_fact_wins(mem):
    mr.add_memory("I live in Paris")
    report = mr.add_memory("I live in Berlin", replaces=1)
    assert "replaces #1" in report and "Paris" in report
    texts = [e["text"] for e in mr.entries()]
    assert texts == ["I live in Berlin"]
    block = mr.read_memory_block("where do I live")
    assert "Berlin" in block and "Paris" not in block
    # A dangling replaces= is reported, not fatal — the new fact still lands.
    assert "no fact #99" in mr.add_memory("I like tea", replaces=99)


def test_replacement_keeps_the_corrected_layer(mem):
    mr.add_memory("Petra wants a weekly summary", layer="entities")
    mr.add_memory("Petra wants a daily summary", replaces=1)  # tool default layer is user
    assert mr.entries()[0]["layer"] == "entities"


def test_duplicate_confirms_and_graduates_inferred(mem):
    mr.add_memory("prefers dark mode", by="inferred")
    assert mr.entries()[0]["by"] == "inferred"
    report = mr.add_memory("Prefers dark mode")  # the user now states it
    assert report.startswith("Already remembered as #1") and "confirmed by you" in report
    e = mr.entries()[0]
    assert e["by"] == "user" and e["n"] == 2
    assert "confirmed ×3" in mr.add_memory("prefers dark mode")


def test_edit_keeps_id_and_provenance(mem):
    mr.add_memory("timezone is PST", run_id=7)
    assert mr.edit_memory(1, "timezone is CET") == "timezone is PST"
    e = mr.entry(1)
    assert e["text"] == "timezone is CET" and e["run"] == 7 and e["n"] == 1
    assert mr.edit_memory(42, "x") is None
    assert mr.edit_memory(1, "   ") is None


# ── the file: hand-editable, migrates, round-trips metadata ─────────────────────────────────


def test_fact_survives_a_new_session_and_metadata_round_trips(mem):
    mr.add_memory("I use Python at work", "work", layer="user", run_id=12, sensitivity=None)
    mr.add_memory("Doctor's appointment Tuesday", layer="commitments", due="2026-09-08",
                  sensitivity="health", by="inferred", run_id=13)
    # A "new session" is a cold parse of the file: nothing is cached in the module.
    raw = mr._read_raw()
    assert "## user" in raw and "## commitments" in raw
    fresh = mr._parse(raw)
    assert [e["text"] for e in fresh] == ["I use Python at work", "Doctor's appointment Tuesday"]
    e = fresh[1]
    assert e == {**e, "id": 2, "layer": "commitments", "by": "inferred", "run": 13,
                 "due": "2026-09-08", "sens": "health", "n": 1, "used": None}
    assert fresh[0]["category"] == "work" and fresh[0]["run"] == 12


def test_legacy_flat_file_reads_as_user_layer_and_migrates_on_write(mem):
    path = mr._memory_path()
    path.parent.mkdir(parents=True)
    path.write_text(
        "# Saturday — persistent memory\n\nold header prose\n\n"
        "- (2026-06-01) [preference] terse answers\n- (2026-06-02) lives in Paris\n",
        encoding="utf-8",
    )
    ents = mr.entries()
    assert [e["layer"] for e in ents] == ["user", "user"]
    # Provisional ids on READ (so /memory forget 2 addresses what the listing showed) — but a
    # read never writes: the file is byte-identical until the next write persists them.
    assert [e["id"] for e in ents] == [1, 2]
    assert "{#" not in mr._read_raw() and "## user" not in mr._read_raw()
    assert mr.remove_memory(2) == "(2026-06-02) lives in Paris"
    assert [e["id"] for e in mr.entries()] == [1]
    mr.add_memory("lives in Paris")
    mr.add_memory("new fact")
    ents = mr.entries()
    assert [e["id"] for e in ents] == [1, 3, 4]  # #2 was forgotten; ids are never reused
    assert "## user" in mr._read_raw()
    assert mr.list_memory()[0] == "(2026-06-01) [preference] terse answers"


def test_hand_written_bullet_without_token_and_braces_in_text(mem):
    mr.add_memory("first")
    path = mr._memory_path()
    path.write_text(path.read_text(encoding="utf-8") + "- hand-written fact\n", encoding="utf-8")
    hand = [e for e in mr.entries() if e["text"] == "hand-written fact"][0]
    assert hand["id"] == 2 and hand["layer"] == "negative"  # appended under the last heading
    # Text that looks like a metadata token is softened so it can't be parsed back as one.
    mr.add_memory("weird {#9 by=user} text")
    ents = {e["text"]: e for e in mr.entries()}
    assert ents["hand-written fact"]["id"] == 2  # the id-less bullet was numbered on that write
    assert ents["weird (#9 by=user) text"]["id"] == 3


# ── selection: what loads, and why ──────────────────────────────────────────────────────────


def test_always_layers_load_whole_and_match_layers_load_by_request(mem):
    mr.add_memory("I prefer terse answers")
    mr.add_memory("send Petra the weekly summary", layer="commitments", due="2026-09-05")
    mr.add_memory("web_extract fails on medium.com — use web_search snippets", layer="agent")
    mr.add_memory("'the deck' means Q3_investor_update.pptx", layer="entities")
    mr.add_memory("do not suggest migrating to Postgres again", layer="negative")

    quiet = mr.read_memory_block("what time is it")
    assert "terse answers" in quiet and "weekly summary" in quiet and "(due 2026-09-05)" in quiet
    assert "medium.com" not in quiet and "deck" not in quiet and "Postgres" not in quiet

    hit = mr.read_memory_block("open the deck and summarize it")
    assert "Q3_investor_update" in hit and "[entities]" in hit
    assert "Postgres" not in hit
    assert "[negative]" in mr.read_memory_block("should we move to postgres?")


def test_memo_digest_is_the_recent_tail_older_memos_by_match(mem):
    for i in range(8):
        mr.add_memory(f"note {i}: decided thing-{i}", layer="memo")
    block = mr.read_memory_block("unrelated request")
    assert "note 7" in block and "note 3" in block and "note 2" not in block
    assert "note 0" in mr.read_memory_block("what was thing-0 about")


def test_cap_omits_and_names_what_did_not_load(mem):
    for i in range(30):
        mr.add_memory(f"user fact number {i} with some padding words to spend the budget")
    sel = mr.select_for_context("", cap=400)
    assert sel["always"] and sel["omitted"] > 0
    block, _ids = mr.memory_context("")
    assert "not loaded" not in block  # the default cap (4000) holds thirty short facts
    small = mr.select_for_context("", cap=400)
    assert small["omitted"] == 30 - len(small["always"])


def test_sensitive_facts_withheld_when_inference_is_remote(mem):
    mr.add_memory("takes blood-pressure medication", sensitivity="health")
    mr.add_memory("likes hiking")
    local = mr.select_for_context("", local_inference=True)
    assert [e["text"] for e in local["always"]] == ["takes blood-pressure medication", "likes hiking"]
    remote = mr.select_for_context("", local_inference=False)
    assert [e["text"] for e in remote["always"]] == ["likes hiking"]
    assert remote["sensitive_withheld"] == 1


def test_mark_used_stamps_matched_only_and_stale_flags(mem):
    mr.add_memory("always loaded", layer="user")
    mr.add_memory("pdf files need /docs add before search", layer="agent")
    block, ids = mr.memory_context("search the pdf report")
    assert "pdf files" in block and ids == [2]
    assert mr.mark_used(ids) == 1
    assert mr.mark_used(ids) == 0  # same day: no rewrite
    assert mr.entry(2)["used"] == str(date.today())
    assert mr.entry(1)["used"] is None
    # Staleness: by-match only, measured from last use (or the learned date).
    assert not mr.is_stale(mr.entry(2))
    far = date.today() + timedelta(days=mr.stale_days() + 1)
    assert mr.is_stale(mr.entry(2), today=far)
    assert not mr.is_stale(mr.entry(1), today=far)


def test_recall_filters_across_layers(mem):
    mr.add_memory("I prefer terse answers")
    mr.add_memory("Petra is my manager", layer="entities")
    assert mr.search_memory("petra") == [f"#2 [entities] ({date.today()}) Petra is my manager"]
    assert mr.search_memory("entities") == mr.search_memory("petra")
    assert len(mr.search_memory("")) == 2
    assert mr.search_memory("nothing here") == []


# ── the grounding node ──────────────────────────────────────────────────────────────────────


def test_grounding_injects_selected_memory_and_stamps_use(mem, monkeypatch):
    from nodes import ground

    mr.add_memory("I prefer terse answers")
    mr.add_memory("'the deck' means Q3_investor_update.pptx", layer="entities")
    stamped = []
    monkeypatch.setattr(ground, "mark_used", lambda ids: stamped.extend(ids))

    state = {"messages": [HumanMessage("summarize the deck")], "current_query": "summarize the deck"}
    context = ground.grounding_node(state)["context"]
    assert "### Persistent memory" in context
    assert "#1 I prefer terse answers" in context
    assert "#2 [entities] 'the deck' means Q3_investor_update.pptx" in context
    assert stamped == [2]

    state = {"messages": [HumanMessage("hello")], "current_query": "hello"}
    context = ground.grounding_node(state)["context"]
    assert "terse answers" in context and "Q3_investor_update" not in context


def test_grounding_omits_section_when_empty_and_profiles_are_gone(mem):
    from nodes import ground

    context = ground.grounding_node({"messages": [], "current_query": "x"})["context"]
    assert "Persistent memory" not in context and "### Profiles" not in context
    assert not hasattr(ground, "_PROFILE_FILES") and not hasattr(ground, "_read_profiles")


# ── the tools ───────────────────────────────────────────────────────────────────────────────


def test_remember_tool_passes_layer_replaces_and_run_provenance(mem, monkeypatch):
    from stores import trace
    from tools.knowledge import recall, remember

    monkeypatch.setattr(trace, "_CURRENT_RUN_ID", 41)
    remember.invoke({"fact": "I live in Paris"})
    out = remember.invoke({"fact": "I live in Berlin", "replaces": 1})
    assert "replaces #1" in out
    remember.invoke({"fact": "Petra is my manager", "layer": "entities"})
    ents = mr.entries()
    assert [(e["text"], e["layer"], e["run"]) for e in ents] == [
        ("I live in Berlin", "user", 41), ("Petra is my manager", "entities", 41)]
    monkeypatch.setattr(trace, "_CURRENT_RUN_ID", None)
    remember.invoke({"fact": "no run"})
    assert mr.entries()[-1]["run"] is None
    assert "Petra" in recall.invoke({"query": "manager"})
    assert recall.invoke({"query": "zzz"}) == "No matching facts in persistent memory."


# ── review fixes (2026-09-02) ───────────────────────────────────────────────────────────────


def test_replaces_retires_the_old_fact_even_when_the_new_text_already_exists(mem):
    mr.add_memory("I live in Paris")
    mr.add_memory("I live in Berlin", by="inferred")
    report = mr.add_memory("I live in Berlin", replaces=1)
    assert "confirmed by you" in report and "replaces #1" in report
    assert [e["text"] for e in mr.entries()] == ["I live in Berlin"]
    assert mr.entry(2)["by"] == "user"


def test_memo_digest_is_stamped_used_and_never_reads_stale(mem):
    old = str(date.today() - timedelta(days=mr.stale_days() + 30))
    mr._memory_path().parent.mkdir(parents=True, exist_ok=True)
    mr._memory_path().write_text(f"## memo\n- ({old}) decided on sqlite {{#1 by=user}}\n")
    assert mr.is_stale(mr.entry(1))  # never loaded yet, learned long ago
    block, ids = mr.memory_context("unrelated request")
    assert "decided on sqlite" in block and ids == [1]  # the digest rides -> it is stamped
    mr.mark_used(ids)
    assert not mr.is_stale(mr.entry(1))


def test_recall_withholds_sensitive_facts_when_inference_is_remote(mem):
    mr.add_memory("takes blood-pressure medication", sensitivity="health")
    mr.add_memory("likes hiking")
    local = mr.search_memory("", local_inference=True)
    assert any("medication" in line for line in local)
    remote = mr.search_memory("", local_inference=False)
    assert not any("medication" in line for line in remote)
    assert remote[-1].startswith("(1 sensitive fact(s) withheld")
    # And the mark is settable from the user-facing paths, not only by hand.
    assert mr.entry(1)["sens"] == "health"


def test_inference_locality_fails_closed(mem, monkeypatch):
    import trust.egress as egress

    monkeypatch.setattr(egress, "ollama_is_local", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert mr._inference_is_local() is False


def test_cap_zero_selects_nothing_and_mark_used_accepts_hash_ids(mem):
    mr.add_memory("a fact")
    mr.add_memory("pdf files need /docs add", layer="agent")
    sel = mr.select_for_context("", cap=0)
    assert sel["always"] == [] and sel["omitted"] == 1
    assert mr.mark_used(["#2"]) == 1 and mr.entry(2)["used"] == str(date.today())


def test_remember_tool_args_survive_coercion_and_hash_ids(mem):
    from core.tool_args import coerce_args
    from tools.knowledge import remember

    args = coerce_args("remember", {"fact": "I live in Paris", "layer": "entities",
                                    "replaces": "#7", "sensitivity": "private", "category": "home"})
    assert args == {"fact": "I live in Paris", "category": "home", "layer": "entities",
                    "replaces": "#7", "sensitivity": "private"}
    remember.invoke({"fact": "I live in Paris"})
    out = remember.invoke({"fact": "I live in Berlin", "replaces": "#1"})  # the spelling the context shows
    assert "replaces #1" in out and [e["text"] for e in mr.entries()] == ["I live in Berlin"]
    out = remember.invoke({"fact": "takes medication", "sensitivity": "health"})
    assert mr.entry(3)["sens"] == "health"
