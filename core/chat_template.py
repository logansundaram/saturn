"""
Raw-mode chat templates — the per-family registry behind interrupt-and-correct continuation.

Autoregressive continuation is *prefix continuation*: to make a model resume a partially-written
(possibly human-edited) assistant turn, the prompt must reproduce the model's chat template
byte-for-byte up to and including the assistant-turn opener, then the edited prefix, and then
STOP — the assistant turn is opened but never closed (no end-of-turn token). The model cannot
tell its own tokens from the user's inside its own prefix, so the continuation is seamless *by
construction* — never by a prompt instruction. A single wrong special token makes the model
restart the turn or emit garbage, so this module is the ONE place raw prompts are rendered
(core/continuation.py routes every continuation through `render_continuation`; nothing else may
hand-roll a template) and `utilities/continuation_contract.py` — the splice-and-continue
contract test — DEFINES which models are supported: a model is supported iff its entry here
passes that test.

── Pinned backend behavior (Ollama 0.30.11, verified live 2026-07-06) ────────────────────────────
- `/api/generate` with `raw: true` sends the prompt through with NO server-side templating.
  The models this registry covers use Ollama's *built-in* renderers (`RENDERER qwen3.5` /
  `RENDERER gemma4` in their Modelfiles — the TEMPLATE field is a bare `{{ .Prompt }}`
  passthrough); raw mode bypasses those renderers too, verified by coherent spliced
  continuations on both families.
- BOS: the gemma4 runner AUTO-ADDS `<bos>` in raw mode (`'hello'` → 2 prompt tokens), and an
  explicit literal `<bos>` is parsed as the special token without doubling (`'<bos>hello'` → 2).
  We therefore render WITHOUT a BOS. qwen3.x adds no BOS at all (`'hello'` → 1) and `<bos>` is
  NOT special there (it tokenizes as text, `'<bos>hello'` → 4) — never emit it.
- `stop` is honored: continuation halts at the family's end-of-turn string with
  `done_reason: "stop"` rather than running to `num_predict`.
- `num_ctx` MUST be set explicitly per request — the daemon otherwise front-truncates silently
  at its own default, which would evict the system prompt from a long continuation.

── Template sources (never hand-guessed) ─────────────────────────────────────────────────────────
- qwen3.x: the GGUF's embedded `tokenizer.chat_template` (read from the qwen3.6:27b blob).
  ChatML turns `<|im_start|>{role}\n…<|im_end|>\n`; the generation opener appends an empty
  think block `<think>\n\n</think>\n\n` (the template's enable_thinking=false form — the
  first-pass /api/chat parser strips thinking from the visible text the user edits, so the
  continuation prefix is post-think prose and the no-think opener is the canonical match).
- gemma4: no embedded GGUF template; the format was recovered from Ollama's built-in `gemma4`
  renderer strings (`<|turn>system\n` / `<|turn>user\n` / `<|turn>model\n`, closed by
  `<turn|>\n`) and verified live by the contract test. Roles are system/user/model.

This module is a leaf: no project imports, so the tests exercise it fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnsupportedModel(ValueError):
    """Raised when a model has no template entry — i.e. continuation is not offered for it.
    Callers treat this as "the freeze feature is off for this model", never as a turn failure."""


@dataclass(frozen=True)
class ChatTemplate:
    """One model family's raw-prompt shape. `render` produces the FULL prompt string with the
    assistant turn opened (assistant_open) and the body appended — and deliberately NOT closed:
    the missing end-of-turn token is what makes the model continue instead of answering afresh."""

    family: str
    prefixes: tuple[str, ...]   # model-id prefixes (lowercase) that select this template
    turn_open: str              # "{role}" -> the turn opener, e.g. "<|im_start|>{role}\n"
    turn_close: str             # end-of-turn suffix for CLOSED turns, e.g. "<|im_end|>\n"
    assistant_open: str         # the generation opener (assistant turn opened, NOT closed)
    assistant_role: str         # what this family calls the assistant ("assistant" / "model")
    stop: tuple[str, ...]       # end-of-turn strings for the request's `stop` parameter

    def render(self, turns: list[tuple[str, str]], assistant_body: str) -> str:
        """The raw continuation prompt: every history turn closed, then the assistant turn
        opened with `assistant_body` and left hanging so generation continues from its last
        character. `turns` are normalized (role, content) pairs — see `normalize_turns`."""
        parts: list[str] = []
        for role, content in turns:
            name = self.assistant_role if role == "assistant" else role
            parts.append(self.turn_open.format(role=name) + content + self.turn_close)
        parts.append(self.assistant_open + assistant_body)
        return "".join(parts)


# The registry. Official support = exactly the families the splice-and-continue contract test
# (utilities/continuation_contract.py) is green on — extend by adding an entry AND running that
# test against the new family; a model absent here simply never arms the freeze hotkey.
TEMPLATES: tuple[ChatTemplate, ...] = (
    ChatTemplate(
        family="qwen3.x",
        prefixes=("qwen3.5", "qwen3.6"),
        turn_open="<|im_start|>{role}\n",
        turn_close="<|im_end|>\n",
        # Empty think block = the template's enable_thinking=false generation opener; the visible
        # prefix being continued is post-think prose (the /api/chat parser strips thinking).
        assistant_open="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        assistant_role="assistant",
        stop=("<|im_end|>",),
    ),
    ChatTemplate(
        family="gemma4",
        prefixes=("gemma4",),
        turn_open="<|turn>{role}\n",
        turn_close="<turn|>\n",
        assistant_open="<|turn>model\n",
        assistant_role="model",
        stop=("<turn|>",),
    ),
)


def template_for(model: str) -> ChatTemplate:
    """The template entry for a model id, matched on name prefix (tags don't matter:
    `qwen3.6:27b` and `qwen3.6:35b` share one shape). Raises UnsupportedModel when absent."""
    name = (model or "").strip().lower()
    for t in TEMPLATES:
        if name.startswith(t.prefixes):
            return t
    raise UnsupportedModel(
        f"no raw-mode chat template for model {model!r} — interrupt-and-correct is only offered "
        f"for families that pass the continuation contract test "
        f"({', '.join(t.family for t in TEMPLATES)}); see core/chat_template.py to extend"
    )


def supported(model: str) -> bool:
    """Whether `model` has a template entry (i.e. the freeze hotkey may arm for it)."""
    try:
        template_for(model)
        return True
    except UnsupportedModel:
        return False


# LangChain message-class name -> template role. Anything unknown is treated as user-role DATA
# (the engine's continuation caller only ever hands system+human context, but a stray message
# must degrade to content-the-model-sees, never crash a resume mid-answer).
_ROLE_BY_TYPE = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "ToolMessage": "user",
}


def normalize_turns(messages: list) -> list[tuple[str, str]]:
    """Flatten LangChain messages (or already-normalized (role, content) pairs) into the
    (role, content) turn list `ChatTemplate.render` consumes. Consecutive same-role messages
    MERGE into one turn (joined by a blank line): the histories the synthesize node builds are
    system + several HumanMessage sections, and one canonical user turn is the shape every
    family was trained on. Empty content drops out."""
    turns: list[tuple[str, str]] = []
    for m in messages or []:
        if isinstance(m, tuple) and len(m) == 2:
            role, content = str(m[0]), str(m[1])
        else:
            role = _ROLE_BY_TYPE.get(type(m).__name__, "user")
            content = getattr(m, "content", "")
            content = content if isinstance(content, str) else str(content)
        content = content.strip()
        if not content:
            continue
        if turns and turns[-1][0] == role:
            turns[-1] = (role, turns[-1][1] + "\n\n" + content)
        else:
            turns.append((role, content))
    return turns


def render_continuation(model: str, messages: list, assistant_body: str) -> str:
    """THE raw continuation prompt builder — every continuation goes through here (single
    render chokepoint; see the module docstring). `assistant_body` is the edited prefix as a
    STRING: the daemon re-tokenizes the whole prompt on every resume, so a character-level edit
    can never produce an invalid token sequence (no token-index surgery anywhere)."""
    return template_for(model).render(normalize_turns(messages), assistant_body)
