# Saturday.ai installer (Windows / PowerShell).
#
#   irm https://raw.githubusercontent.com/logansundaram/saturn/main/install.ps1 | iex
#
# Local-first install: clones the repo, builds an isolated venv, installs Ollama if missing,
# pulls the local models, and puts a `saturn` command on your PATH. Re-running updates an
# existing install in place. Read before you pipe - it's meant to be.

$ErrorActionPreference = 'Stop'

# --- config (override via env) -----------------------------------------------------
$RepoUrl    = if ($env:SATURDAY_REPO)   { $env:SATURDAY_REPO }   else { 'https://github.com/logansundaram/saturn.git' }
$Branch     = if ($env:SATURDAY_BRANCH) { $env:SATURDAY_BRANCH } else { 'main' }
$InstallDir = if ($env:SATURDAY_HOME)   { $env:SATURDAY_HOME }   else { Join-Path $env:USERPROFILE '.saturday' }
$BinDir     = if ($env:SATURDAY_BIN)    { $env:SATURDAY_BIN }    else { Join-Path $InstallDir 'bin' }
# Active tier for a fresh install. 'laptop' uses the small gemma4 models (light download, runs on
# modest hardware); switch to 'workstation'/'cloud-hybrid' later via /models.
$Tier       = if ($env:SATURDAY_TIER)   { $env:SATURDAY_TIER }   else { 'laptop' }
# Local models the laptop tier needs (small gemma4 chat model + the RAG embedder).
$Models     = if ($env:SATURDAY_MODELS) { $env:SATURDAY_MODELS -split '\s+' } else { @('gemma4:e4b', 'qwen3-embedding:8b') }

# --- output helpers ----------------------------------------------------------------
function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "  ok $m" -ForegroundColor Green }
function Warn ($m) { Write-Host " warn $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host "error $m" -ForegroundColor Red; exit 1 }
function Have ($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }

# --- 1. prerequisites --------------------------------------------------------------
Say 'Checking prerequisites'
if (-not (Have git)) { Die 'git is required. Install it (https://git-scm.com), then re-run.' }

$Py = $null
foreach ($c in @('py', 'python', 'python3')) {
  if (Have $c) {
    & $c -c 'import sys;exit(0 if sys.version_info>=(3,10) else 1)' 2>$null
    if ($LASTEXITCODE -eq 0) { $Py = $c; break }
  }
}
if (-not $Py) { Die 'Python 3.10+ is required and was not found. Install it from https://python.org, then re-run.' }
Ok ("Python: " + (& $Py --version 2>&1))

# --- 2. Ollama (local model runtime) ----------------------------------------------
if (Have ollama) {
  Ok 'Ollama already installed'
} else {
  Say 'Installing Ollama'
  if (Have winget) {
    winget install --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [Environment]::GetEnvironmentVariable('Path','User')
  } else {
    Die 'Ollama not found and winget is unavailable. Install Ollama from https://ollama.com/download, then re-run.'
  }
  if (-not (Have ollama)) { Die 'Ollama install did not complete. Install from https://ollama.com/download and re-run.' }
}

# Make sure the daemon answers before we pull.
& ollama list *> $null
if ($LASTEXITCODE -ne 0) {
  Say 'Starting the Ollama daemon'
  Start-Process -WindowStyle Hidden ollama -ArgumentList 'serve' -ErrorAction SilentlyContinue
  for ($i = 0; $i -lt 30; $i++) { & ollama list *> $null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 1 }
}
& ollama list *> $null
if ($LASTEXITCODE -ne 0) { Warn "Could not reach the Ollama daemon - model pulls may fail. Start it ('ollama serve') and re-run." }

# --- 3. clone or update the repo ---------------------------------------------------
if (Test-Path (Join-Path $InstallDir '.git')) {
  Say "Updating existing install at $InstallDir"
  git -C $InstallDir fetch --quiet origin $Branch
  git -C $InstallDir checkout --quiet $Branch
  git -C $InstallDir pull --quiet --ff-only origin $Branch
  if ($LASTEXITCODE -ne 0) { Warn 'Could not fast-forward (local changes?) - keeping current checkout.' }
} else {
  Say "Cloning Saturday.ai into $InstallDir"
  git clone --quiet --branch $Branch $RepoUrl $InstallDir
}
Ok 'Source ready'

# --- 4. isolated environment + dependencies ----------------------------------------
Say 'Creating virtual environment and installing dependencies'
$VenvPy = Join-Path $InstallDir '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { & $Py -m venv (Join-Path $InstallDir '.venv') }
& $VenvPy -m pip install --quiet --upgrade pip
# Not --quiet: a swallowed dependency failure here is how a broken install slips through.
& $VenvPy -m pip install -r (Join-Path $InstallDir 'requirements.txt')
Ok 'Dependencies installed'

# --- 4b. select the active tier (in-place, comment-preserving) ----------------------
Say "Setting active tier to '$Tier'"
Push-Location $InstallDir
try {
  & $VenvPy -c "import sys; from config import get_config, persist; get_config().set('active_tier', sys.argv[1]); persist('active_tier')" $Tier
  if ($LASTEXITCODE -eq 0) { Ok "Tier set to '$Tier'" } else { Warn 'Could not set tier - defaulting to config.yaml. Change it later with /models.' }
} finally { Pop-Location }

# --- 5. pull local models ----------------------------------------------------------
if ($Models.Count -gt 0) {
  # Show ollama's live progress (no redirect) - multi-GB downloads, a silent pull looks frozen.
  Say 'Pulling local models (several GB - live progress below)'
  foreach ($m in $Models) {
    Say "  $m"
    & ollama pull $m
    if ($LASTEXITCODE -eq 0) { Ok $m }
    else { Warn "pull '$m' failed - pull it later with: ollama pull $m" }
  }
}

# --- 6. launcher on PATH -----------------------------------------------------------
Say "Installing the 'saturn' launcher into $BinDir"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
$launcher = @"
@echo off
REM Saturday.ai launcher - runs the agent from its isolated venv.
"$VenvPy" "$(Join-Path $InstallDir 'agent.py')" %*
"@
Set-Content -Path (Join-Path $BinDir 'saturn.cmd') -Value $launcher -Encoding ascii
Ok 'Launcher installed'

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if ($userPath -notlike "*$BinDir*") {
  [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $BinDir), 'User')
  Warn "Added $BinDir to your PATH - open a new terminal for 'saturn' to resolve."
}

# --- done --------------------------------------------------------------------------
Write-Host ''
Ok 'Saturday.ai installed.'
Write-Host "Run: saturn   (open a new terminal if PATH was just updated)" -ForegroundColor Cyan
Write-Host 'First launch runs a setup check (/config setup). Use ''saturn -p "your question"'' for one-shot mode.'
