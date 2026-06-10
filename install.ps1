# Saturn installer (Windows / PowerShell) - by Saturday.ai.
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
# Local models the laptop tier needs (small gemma4 chat model + the RAG embedder). Must match the
# `laptop` tier bindings in config.yaml - pulling different models than the tier binds breaks the
# first run. If you override this, rebind the roles afterwards with /models.
$Models     = if ($env:SATURDAY_MODELS) { $env:SATURDAY_MODELS -split '\s+' } else { @('gemma4:e4b', 'qwen3-embedding:8b') }
# Minimum Ollama daemon version. Older daemons can't pull the current model formats (the pull
# fails or the model runs wrong), so we update below if the installed one is behind this.
$MinOllama  = if ($env:SATURDAY_MIN_OLLAMA) { $env:SATURDAY_MIN_OLLAMA } else { '0.6.0' }

# --- output helpers ----------------------------------------------------------------
function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "  ok $m" -ForegroundColor Green }
function Warn ($m) { Write-Host " warn $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host "error $m" -ForegroundColor Red; exit 1 }
function Have ($c) { [bool](Get-Command $c -ErrorAction SilentlyContinue) }
# True if dotted-numeric version $a is older than $b (e.g. VerLt '0.5.9' '0.6.0' -> $true).
function VerLt ($a, $b) {
  $av = @($a -split '\.' | ForEach-Object { [int]($_ -replace '\D', '') })
  $bv = @($b -split '\.' | ForEach-Object { [int]($_ -replace '\D', '') })
  for ($i = 0; $i -lt [math]::Max($av.Count, $bv.Count); $i++) {
    $x = if ($i -lt $av.Count) { $av[$i] } else { 0 }
    $y = if ($i -lt $bv.Count) { $bv[$i] } else { 0 }
    if ($x -lt $y) { return $true }
    if ($x -gt $y) { return $false }
  }
  return $false
}

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
if (-not $Py) { Die 'Python 3.10+ is required (older versions cannot run Saturn). Install or update it from https://python.org, then re-run.' }
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

# Ollama must be recent enough to pull the current model formats - an old daemon fails the pull.
$ollVer = ''
try { if ((& ollama --version 2>&1) -match '(\d+\.\d+\.\d+)') { $ollVer = $matches[1] } } catch {}
if ($ollVer -and (VerLt $ollVer $MinOllama)) {
  Warn "Ollama $ollVer is older than $MinOllama and may fail to pull the local models."
  if (Have winget) {
    Say 'Updating Ollama'
    winget upgrade --id Ollama.Ollama -e --silent --accept-source-agreements --accept-package-agreements
    $ollVer = ''
    try { if ((& ollama --version 2>&1) -match '(\d+\.\d+\.\d+)') { $ollVer = $matches[1] } } catch {}
    if ($ollVer -and (VerLt $ollVer $MinOllama)) { Warn "Still on Ollama $ollVer - update manually from https://ollama.com/download if model pulls fail." }
    elseif ($ollVer) { Ok "Ollama updated to $ollVer" }
  } else {
    Warn 'Update it from https://ollama.com/download before pulling models.'
  }
} elseif ($ollVer) {
  Ok "Ollama $ollVer"
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
  Say "Cloning Saturn into $InstallDir"
  git clone --quiet --branch $Branch $RepoUrl $InstallDir
}
Ok 'Source ready'

# --- 4. isolated environment + dependencies ----------------------------------------
Say 'Creating virtual environment'
$VenvPy = Join-Path $InstallDir '.venv\Scripts\python.exe'
if (-not (Test-Path $VenvPy)) { & $Py -m venv (Join-Path $InstallDir '.venv') }
& $VenvPy -m pip install --quiet --upgrade pip

# Install requirements behind a compact progress bar instead of pip's full firehose. Everything
# is still captured to $pipLog so a dependency failure is never silently swallowed.
Say 'Installing dependencies'
$reqFile = Join-Path $InstallDir 'requirements.txt'
$pipLog  = Join-Path $InstallDir '.venv\pip-install.log'
Remove-Item $pipLog -ErrorAction SilentlyContinue
# Rough upper bound for the bar: declared requirements + headroom for transitive deps.
$expected = [math]::Max(1, ((Get-Content $reqFile | Where-Object { $_ -match '^\s*[^#\s]' }).Count) * 3)
$seen = 0
& {
  $ErrorActionPreference = 'Continue'   # pip writes progress/warnings to stderr; don't let that abort us
  & $VenvPy -m pip install --no-input --progress-bar off -r $reqFile 2>&1
} | ForEach-Object {
  Add-Content -LiteralPath $pipLog -Value ([string]$_)
  if ([string]$_ -match '^\s*Collecting\s+([^\s=<>!~;\[]+)') {
    $seen++
    $pct = [math]::Min(95, [int](($seen / $expected) * 100))
    Write-Progress -Activity 'Installing dependencies' -Status $matches[1] -PercentComplete $pct
  }
}
$pipExit = $LASTEXITCODE
Write-Progress -Activity 'Installing dependencies' -Completed
if ($pipExit -ne 0) {
  Warn 'Dependency install failed - last lines of pip output:'
  if (Test-Path $pipLog) { Get-Content $pipLog -Tail 30 | ForEach-Object { Write-Host "    $_" } }
  Die "Could not install Python dependencies. Full log: $pipLog"
}
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
REM Saturn launcher - runs the agent from its isolated venv.
"$VenvPy" "$(Join-Path $InstallDir 'agent.py')" %*
"@
# Default (ANSI) encoding, not ascii: a username with non-ASCII characters would corrupt the
# venv path inside the launcher if forced to 7-bit.
Set-Content -Path (Join-Path $BinDir 'saturn.cmd') -Value $launcher -Encoding default
Ok 'Launcher installed'

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }   # a fresh account can have an empty user PATH
if ($userPath -notlike "*$BinDir*") {
  $newPath = if ($userPath) { $userPath.TrimEnd(';') + ';' + $BinDir } else { $BinDir }
  [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
  Warn "Added $BinDir to your PATH - open a new terminal for 'saturn' to resolve."
}

# --- done --------------------------------------------------------------------------
Write-Host ''
Ok 'Saturn installed.'
Write-Host "Run: saturn   (open a new terminal if PATH was just updated)" -ForegroundColor Cyan
Write-Host 'First launch runs a setup check (/config setup). Use ''saturn -p "your question"'' for one-shot mode.'
