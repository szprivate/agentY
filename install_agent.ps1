#Requires -Version 7
<#
.SYNOPSIS
    Cross-platform install script for the agentY project.
    Works on Windows and macOS (pwsh 7+).
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "━━━  $Text  ━━━" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Text)
    Write-Host "  ✔  $Text" -ForegroundColor Green
}

function Write-Info {
    param([string]$Text)
    Write-Host "  ℹ  $Text" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Text)
    Write-Host "  ✖  $Text" -ForegroundColor Red
}

function Exit-WithError {
    param([string]$Message, [int]$Code = 1)
    Write-Fail $Message
    exit $Code
}

# Replace or append a KEY=VALUE line in a file.
function Set-EnvKey {
    param(
        [string]$FilePath,
        [string]$Key,
        [string]$Value
    )
    $content = Get-Content $FilePath -Raw
    if ($null -eq $content) { $content = "" }

    # Escape the key for regex use
    $escapedKey = [regex]::Escape($Key)
    $pattern    = "(?m)^$escapedKey=.*$"
    $replacement = "$Key=$Value"

    if ($content -match $pattern) {
        $content = $content -replace $pattern, $replacement
    } else {
        # Ensure file ends with a newline before appending
        if ($content.Length -gt 0 -and -not $content.EndsWith("`n")) {
            $content += "`n"
        }
        $content += "$replacement`n"
    }

    # Write without BOM, unix line endings inside the file are fine
    [System.IO.File]::WriteAllText($FilePath, $content, [System.Text.UTF8Encoding]::new($false))
}

# ---------------------------------------------------------------------------
# Resolve script / project root
# ---------------------------------------------------------------------------
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = (Get-Location).Path
}

# ---------------------------------------------------------------------------
# 1. Preflight checks
# ---------------------------------------------------------------------------

Write-Header "1 / 8  Preflight checks"

# uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Exit-WithError @"
'uv' is not installed or not on PATH.
Install it from: https://docs.astral.sh/uv/getting-started/installation/
"@
}
Write-Success "uv found: $(uv --version)"

# Docker CLI
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Exit-WithError "Docker is not installed or not on PATH. Install Docker Desktop and try again."
}
Write-Success "Docker CLI found"

# Docker daemon running
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw }
} catch {
    Exit-WithError "Docker daemon is not running. Start Docker Desktop (or the Docker service) and try again."
}
Write-Success "Docker daemon is running"

# Node.js (needed for npx prisma)
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Exit-WithError "Node.js is not installed or not on PATH. Install it from https://nodejs.org/ and try again."
}
Write-Success "Node.js found: $(node --version)"

# docker-compose.yml
$ComposeFile = Join-Path $ProjectRoot "docker-compose.yml"
if (-not (Test-Path $ComposeFile)) {
    Exit-WithError "docker-compose.yml not found in project root: $ProjectRoot"
}
Write-Success "docker-compose.yml found"

# ---------------------------------------------------------------------------
# 2. Virtual environment
# ---------------------------------------------------------------------------

Write-Header "2 / 8  Virtual environment"

$VenvDir = Join-Path $ProjectRoot ".venv"

if (-not (Test-Path $VenvDir)) {
    Write-Info "Creating .venv with uv …"
    Push-Location $ProjectRoot
    uv venv .venv
    if ($LASTEXITCODE -ne 0) { Exit-WithError "uv venv failed." }
    Pop-Location
    Write-Success ".venv created"
} else {
    Write-Info ".venv already exists — skipping creation"
}

# Activate (cross-platform)
if ($IsWindows) {
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
} else {
    $ActivateScript = Join-Path $VenvDir "bin/Activate.ps1"
}

if (-not (Test-Path $ActivateScript)) {
    Exit-WithError "Could not find activation script at: $ActivateScript"
}

Write-Info "Activating .venv …"
& $ActivateScript
Write-Success ".venv activated"

# ---------------------------------------------------------------------------
# 3. Python dependencies
# ---------------------------------------------------------------------------

Write-Header "3 / 8  Python dependencies"

Push-Location $ProjectRoot

Write-Info "Installing requirements.txt …"
uv pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Exit-WithError "uv pip install -r requirements.txt failed." }

Write-Info "Installing asyncpg and boto3 …"
uv pip install asyncpg boto3
if ($LASTEXITCODE -ne 0) { Exit-WithError "uv pip install asyncpg boto3 failed." }

Pop-Location
Write-Success "Python dependencies installed"

# ---------------------------------------------------------------------------
# 4. .env setup
# ---------------------------------------------------------------------------

Write-Header "4 / 8  .env setup"

$EnvFile    = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env_example"

if (-not (Test-Path $EnvFile)) {
    if (-not (Test-Path $EnvExample)) {
        Exit-WithError ".env_example not found in project root. Cannot create .env."
    }
    Copy-Item $EnvExample $EnvFile
    Write-Info "Copied .env_example → .env"
    Write-Host ""
    Write-Host "  ⚠  ACTION REQUIRED: Open .env and fill in your API keys before running the agent." -ForegroundColor Magenta
} else {
    Write-Info ".env already exists — skipping copy"
}

# ---------------------------------------------------------------------------
# 5. Chainlit auth setup
# ---------------------------------------------------------------------------

Write-Header "5 / 8  Chainlit auth setup"

# Username
Write-Host ""
$ChainlitUsername = Read-Host "  Enter Chainlit username"
if ([string]::IsNullOrWhiteSpace($ChainlitUsername)) {
    Exit-WithError "Username cannot be empty."
}

# Password (with confirmation loop)
while ($true) {
    $Password1 = Read-Host "  Enter Chainlit password" -AsSecureString
    $Password2 = Read-Host "  Confirm Chainlit password" -AsSecureString

    # Convert SecureString → plain text for comparison
    $Plain1 = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password1))
    $Plain2 = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password2))

    if ($Plain1 -ceq $Plain2) {
        $ChainlitPassword = $Plain1
        break
    }
    Write-Fail "Passwords do not match. Please try again."
}

# Generate Chainlit auth secret
Write-Info "Generating Chainlit auth secret …"
Push-Location $ProjectRoot
$SecretOutput = chainlit create-secret 2>&1
Pop-Location

if ($LASTEXITCODE -ne 0) {
    Exit-WithError "chainlit create-secret failed. Is chainlit installed? ($SecretOutput)"
}

# Parse: find a line containing CHAINLIT_AUTH_SECRET=
$SecretLine = ($SecretOutput | Where-Object { $_ -match "CHAINLIT_AUTH_SECRET\s*=" }) | Select-Object -First 1
if (-not $SecretLine) {
    Exit-WithError "Could not parse CHAINLIT_AUTH_SECRET from chainlit create-secret output.`nOutput was:`n$SecretOutput"
}
$ChainlitSecret = ($SecretLine -replace "^.*CHAINLIT_AUTH_SECRET\s*=\s*", "").Trim()

# Write to .env
Set-EnvKey -FilePath $EnvFile -Key "CHAINLIT_AUTH_SECRET" -Value $ChainlitSecret
Set-EnvKey -FilePath $EnvFile -Key "CHAINLIT_USERNAME"    -Value $ChainlitUsername
Set-EnvKey -FilePath $EnvFile -Key "CHAINLIT_PASSWORD"    -Value $ChainlitPassword

Write-Success "Chainlit auth credentials written to .env"

# ---------------------------------------------------------------------------
# 6. Docker services
# ---------------------------------------------------------------------------

Write-Header "6 / 8  Docker services (MinIO + PostgreSQL)"

Push-Location $ProjectRoot
Write-Info "Starting containers with docker compose up -d …"
docker compose up -d
if ($LASTEXITCODE -ne 0) { Exit-WithError "docker compose up -d failed." }
Pop-Location

# Wait for MinIO
Write-Info "Waiting for MinIO to be healthy …"
$MinioUrl  = "http://localhost:9000/minio/health/live"
$MaxWait   = 30
$Elapsed   = 0
$MinioOk   = $false

while ($Elapsed -lt $MaxWait) {
    try {
        $response = Invoke-WebRequest -Uri $MinioUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            $MinioOk = $true
            break
        }
    } catch { }
    Write-Host "." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
    $Elapsed++
}
Write-Host ""

if (-not $MinioOk) {
    Exit-WithError "MinIO did not become healthy within ${MaxWait}s. Check `docker compose logs minio`."
}
Write-Success "MinIO is healthy"

# Wait for PostgreSQL
Write-Info "Waiting for PostgreSQL to be ready …"
$Elapsed  = 0
$PgOk     = $false

while ($Elapsed -lt $MaxWait) {
    Push-Location $ProjectRoot
    $pgOut = docker compose exec -T postgres pg_isready 2>&1
    $pgExit = $LASTEXITCODE
    Pop-Location

    if ($pgExit -eq 0) {
        $PgOk = $true
        break
    }
    Write-Host "." -NoNewline -ForegroundColor DarkGray
    Start-Sleep -Seconds 1
    $Elapsed++
}
Write-Host ""

if (-not $PgOk) {
    Exit-WithError "PostgreSQL did not become ready within ${MaxWait}s. Check `docker compose logs postgres`."
}
Write-Success "PostgreSQL is ready"

# ---------------------------------------------------------------------------
# 7. Prisma migration
# ---------------------------------------------------------------------------

Write-Header "7 / 8  Prisma migration"

Push-Location $ProjectRoot
Write-Info "Running npx prisma migrate deploy …"
npx prisma migrate deploy
$PrismaExit = $LASTEXITCODE
Pop-Location

if ($PrismaExit -ne 0) {
    Exit-WithError "npx prisma migrate deploy failed (exit code $PrismaExit). Check the output above for details."
}
Write-Success "Prisma migrations applied"

# ---------------------------------------------------------------------------
# 8. Final summary
# ---------------------------------------------------------------------------

Write-Header "8 / 8  Setup complete"
Write-Host ""
Write-Host "  agentY is ready to go!  Here's what was set up:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  • Python .venv created and dependencies installed" -ForegroundColor White
Write-Host "  • .env created and Chainlit auth credentials written" -ForegroundColor White
Write-Host "  • MinIO and PostgreSQL containers are running" -ForegroundColor White
Write-Host "  • Prisma migrations applied" -ForegroundColor White
Write-Host ""
Write-Host "  ─── Next steps ──────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  1. Fill in remaining API keys in .env before the first run" -ForegroundColor Yellow
Write-Host "     (HF_TOKEN, ANTHROPIC_API_KEY, COMFYUI_API_KEY, etc.)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  2. Start the agent:" -ForegroundColor Yellow
Write-Host "     chainlit run src/chainlit_app.py" -ForegroundColor White
Write-Host ""
Write-Host "  3. MinIO console:  http://localhost:9001" -ForegroundColor Yellow
Write-Host "     Credentials:    minioadmin / minioadmin" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  4. Prisma Studio:  npx prisma studio" -ForegroundColor Yellow
Write-Host ""
Write-Host "  ─────────────────────────────────────────────────────────────" -ForegroundColor DarkGray
Write-Host ""
