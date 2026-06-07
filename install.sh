#!/bin/sh
# Saturday.ai installer (macOS / Linux).
#
#   curl -fsSL https://raw.githubusercontent.com/logansundaram/saturn/main/install.sh | sh
#
# Local-first install: clones the repo, builds an isolated venv, installs Ollama if
# missing, pulls the local models, and drops a `saturn` launcher on your PATH.
# Re-running updates an existing install in place. Read before you pipe - it's meant to be.
set -eu

# --- config (override via env) -----------------------------------------------------
REPO_URL="${SATURDAY_REPO:-https://github.com/logansundaram/saturn.git}"
BRANCH="${SATURDAY_BRANCH:-main}"
INSTALL_DIR="${SATURDAY_HOME:-$HOME/.saturday}"
BIN_DIR="${SATURDAY_BIN:-$HOME/.local/bin}"
MIN_PY_MAJOR=3
MIN_PY_MINOR=10
# Active tier for a fresh install. 'laptop' uses the small gemma4 models so the download is
# light and it runs on modest hardware; switch to 'workstation'/'cloud-hybrid' later via /models.
TIER="${SATURDAY_TIER:-laptop}"
# Local models the laptop tier needs (small gemma4 chat model + the RAG embedder). Override to
# skip/customize, e.g. SATURDAY_MODELS="gemma4:e2b qwen3-embedding:8b" for an even lighter chat model.
MODELS="${SATURDAY_MODELS:-gemma4:e4b qwen3-embedding:8b}"

# --- output helpers ----------------------------------------------------------------
if [ -t 1 ]; then B="$(printf '\033[1m')"; G="$(printf '\033[32m')"; Y="$(printf '\033[33m')"; R="$(printf '\033[31m')"; X="$(printf '\033[0m')"; else B=; G=; Y=; R=; X=; fi
say()  { printf '%s==>%s %s\n' "$B" "$X" "$1"; }
ok()   { printf '%s  ok%s %s\n' "$G" "$X" "$1"; }
warn() { printf '%s warn%s %s\n' "$Y" "$X" "$1"; }
die()  { printf '%serror%s %s\n' "$R" "$X" "$1" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. prerequisites --------------------------------------------------------------
say "Checking prerequisites"
have git || die "git is required. Install it, then re-run."

PY=""
for c in python3 python; do
  if have "$c" && "$c" -c "import sys;exit(0 if sys.version_info>=($MIN_PY_MAJOR,$MIN_PY_MINOR) else 1)" 2>/dev/null; then PY="$c"; break; fi
done
[ -n "$PY" ] || die "Python ${MIN_PY_MAJOR}.${MIN_PY_MINOR}+ is required and was not found."
ok "Python: $("$PY" --version 2>&1) ($(command -v "$PY"))"

# --- 2. Ollama (local model runtime) ----------------------------------------------
if have ollama; then
  ok "Ollama already installed"
else
  say "Installing Ollama"
  OS="$(uname -s)"
  if [ "$OS" = "Linux" ]; then
    curl -fsSL https://ollama.com/install.sh | sh
  elif [ "$OS" = "Darwin" ]; then
    if have brew; then brew install --cask ollama || brew install ollama
    else die "Ollama not found. Install it from https://ollama.com/download (or 'brew install ollama'), then re-run."; fi
  else
    die "Unsupported OS '$OS' for auto-install. Install Ollama from https://ollama.com/download, then re-run."
  fi
  have ollama || die "Ollama install did not complete. Install from https://ollama.com/download and re-run."
fi

# Make sure the daemon answers before we try to pull. On Linux the installer sets up a
# service; if it's not up (or on macOS without the app running) start it in the background.
if ! ollama list >/dev/null 2>&1; then
  say "Starting the Ollama daemon"
  (ollama serve >/dev/null 2>&1 &) || true
  i=0; while [ "$i" -lt 30 ]; do ollama list >/dev/null 2>&1 && break; i=$((i+1)); sleep 1; done
fi
ollama list >/dev/null 2>&1 || warn "Could not reach the Ollama daemon - model pulls below may fail. Start it with 'ollama serve' and re-run."

# --- 3. clone or update the repo ---------------------------------------------------
if [ -d "$INSTALL_DIR/.git" ]; then
  say "Updating existing install at $INSTALL_DIR"
  git -C "$INSTALL_DIR" fetch --quiet origin "$BRANCH"
  git -C "$INSTALL_DIR" checkout --quiet "$BRANCH"
  git -C "$INSTALL_DIR" pull --quiet --ff-only origin "$BRANCH" || warn "Could not fast-forward (local changes?) - keeping current checkout."
else
  say "Cloning Saturday.ai into $INSTALL_DIR"
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
ok "Source ready"

# --- 4. isolated environment + dependencies ----------------------------------------
say "Creating virtual environment and installing dependencies"
[ -d "$INSTALL_DIR/.venv" ] || "$PY" -m venv "$INSTALL_DIR/.venv"
VENV_PY="$INSTALL_DIR/.venv/bin/python"
"$VENV_PY" -m pip install --quiet --upgrade pip
# Not --quiet: a swallowed dependency failure here is how a broken install slips through
# (e.g. a missing package). Let pip's output and any error be visible.
"$VENV_PY" -m pip install -r "$INSTALL_DIR/requirements.txt"
ok "Dependencies installed"

# --- 4b. select the active tier (in-place, comment-preserving) ----------------------
say "Setting active tier to '$TIER'"
if ( cd "$INSTALL_DIR" && "$VENV_PY" - "$TIER" <<'PYEOF'
import sys
from config import get_config, persist
get_config().set("active_tier", sys.argv[1])
persist("active_tier")
PYEOF
); then ok "Tier set to '$TIER'"; else warn "Could not set tier - defaulting to config.yaml. Change it later with /models."; fi

# --- 5. pull local models ----------------------------------------------------------
# Show ollama's live progress bar (do NOT redirect to /dev/null) - these are multi-GB
# downloads and a silent pull looks like a frozen installer.
if [ -n "$MODELS" ]; then
  say "Pulling local models (several GB - live progress below)"
  for m in $MODELS; do
    say "  $m"
    if ollama pull "$m"; then ok "$m"; else warn "pull '$m' failed - pull it later with: ollama pull $m"; fi
  done
fi

# --- 6. launcher on PATH -----------------------------------------------------------
say "Installing the 'saturn' launcher into $BIN_DIR"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/saturn" <<EOF
#!/bin/sh
# Saturday.ai launcher - runs the agent from its isolated venv.
exec "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/agent.py" "\$@"
EOF
chmod +x "$BIN_DIR/saturn"
ok "Launcher installed"

# --- 7. ensure the launcher is on PATH ---------------------------------------------
# Pick the shell rc file we'd persist PATH to.
case "$(basename "${SHELL:-sh}")" in
  zsh)  PROFILE="$HOME/.zshrc" ;;
  bash) [ "$(uname -s)" = "Darwin" ] && PROFILE="$HOME/.bash_profile" || PROFILE="$HOME/.bashrc" ;;
  *)    PROFILE="$HOME/.profile" ;;
esac

# Append the export line to $PROFILE (idempotent - safe on re-runs).
add_to_path() {
  LINE="export PATH=\"$BIN_DIR:\$PATH\""
  grep -qsF "$LINE" "$PROFILE" 2>/dev/null || printf '\n# Added by the Saturday.ai installer\n%s\n' "$LINE" >> "$PROFILE"
  ok "Added $BIN_DIR to PATH in $PROFILE"
  printf '%sActivate it now:%s export PATH="%s:$PATH"   (or just open a new terminal)\n' "$B" "$X" "$BIN_DIR"
}

echo
ok "Saturday.ai installed."
case ":$PATH:" in
  *":$BIN_DIR:"*)
    printf '%sRun:%s saturn\n' "$B" "$X" ;;
  *)
    warn "$BIN_DIR is not on your PATH."
    # stdin is the piped script under `curl | sh`, so ask on the controlling terminal.
    # If we can't open one (headless/CI), skip the prompt and add it by default.
    ans=y
    if { exec 3</dev/tty; } 2>/dev/null; then
      printf ' Add it to %s automatically? [Y/n] ' "$PROFILE"
      read ans <&3 || ans=y
      exec 3<&-
    fi
    case "${ans:-y}" in
      [Nn]*)
        warn "Skipped. Add it later with:"
        printf '       echo '\''export PATH="%s:$PATH"'\'' >> %s\n' "$BIN_DIR" "$PROFILE"
        printf '     Until then, run: %s/saturn\n' "$BIN_DIR" ;;
      *)
        add_to_path ;;
    esac ;;
esac
echo "First launch runs a setup check (/config setup). Use 'saturn -p \"your question\"' for one-shot mode."
