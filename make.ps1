<#
.SYNOPSIS
  Windows shim for the Makefile targets (GNU make is not installed by default
  on Windows). Keep the two in sync.

.EXAMPLE
  .\make.ps1 setup
  .\make.ps1 test
  .\make.ps1 eval -Job 04
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [string]$Job,
    [string]$Service
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Step {
    param([string]$Exe, [string[]]$CmdArgs)
    Write-Host ">> $Exe $($CmdArgs -join ' ')" -ForegroundColor DarkGray
    & $Exe @CmdArgs
    if ($LASTEXITCODE -ne 0) { throw "$Exe exited with $LASTEXITCODE" }
}

function Uvr { param([string[]]$CmdArgs) Invoke-Step 'uv' (@('run') + $CmdArgs) }
function Compose { param([string[]]$CmdArgs) Invoke-Step 'docker' (@('compose') + $CmdArgs) }

switch ($Target.ToLower()) {
    'help' {
        Write-Host @'
secai targets (Windows shim for the Makefile)

  setup              create the venv, install deps, install pre-commit hooks
  secrets            print freshly generated Langfuse server secrets
  langfuse-auth      print LANGFUSE_AUTH (base64 of the key pair) from .env
  env-check          check .env still matches the running stack
  env-restore        rebuild .env from the running containers

  up                 start the full stack (includes vLLM; needs an NVIDIA GPU)
  pull-llm           pull the pinned vLLM image (large; foreground)
  up-llm             start vLLM on its own, on top of a running core stack
  down-llm           stop vLLM, releasing the GPU and its RAM
  up-core            start everything except vLLM (no GPU required)
  down               stop the stack (volumes preserved)
  restart            down + up
  ps                 service status
  logs               tail logs        (-Service <name> to narrow)

  test               unit suite with coverage
  test-unit          unit suite only
  test-integration   integration suite against the running stack

  lint               ruff check + format --check
  fmt                ruff format + --fix
  typecheck          mypy --strict on src
  check              lint + typecheck + test

  eval               gold-set evaluation    (-Job 04)
  clean              remove caches and build artefacts
'@
    }

    'setup' {
        Invoke-Step 'uv' @('sync', '--all-groups')
        if (-not (Test-Path '.env')) {
            Copy-Item '.env.example' '.env'
            Write-Host 'created .env from .env.example - fill in the secrets' -ForegroundColor Yellow
        }
        try { Uvr @('pre-commit', 'install') } catch { Write-Warning 'pre-commit hook install skipped' }
    }

    'secrets'       { Uvr @('python', 'scripts/gen_secrets.py') }
    'langfuse-auth' { Uvr @('python', 'scripts/langfuse_auth.py') }
    'env-check'     { Uvr @('python', 'scripts/restore_env.py', '--check') }
    'env-restore'   { Uvr @('python', 'scripts/restore_env.py') }

    'up'      { Compose @('--profile', 'llm', 'up', '-d', '--wait') }
    'up-core' { Compose @('up', '-d', '--wait', 'postgres', 'langfuse-web', 'langfuse-worker', 'otel-collector') }
    'pull-llm' { Compose @('--profile', 'llm', 'pull', 'vllm') }
    'up-llm'   { Compose @('--profile', 'llm', 'up', '-d', '--wait', 'vllm') }
    'down-llm' { Compose @('--profile', 'llm', 'rm', '-sf', 'vllm') }
    'down'    { Compose @('down') }
    'restart' { Compose @('down'); Compose @('--profile', 'llm', 'up', '-d', '--wait') }
    'ps'      { Compose @('ps') }
    'logs'    {
        $a = @('logs', '-f', '--tail=100')
        if ($Service) { $a += $Service }
        Compose $a
    }

    'test'             { Uvr @('pytest', 'tests/unit', '--cov', '--cov-report=term-missing', '--cov-report=xml') }
    'test-unit'        { Uvr @('pytest', 'tests/unit') }
    'test-integration' { Uvr @('pytest', 'tests/integration', '-m', 'integration') }

    'lint' {
        Uvr @('ruff', 'check', 'src', 'tests', 'scripts')
        Uvr @('ruff', 'format', '--check', 'src', 'tests', 'scripts')
    }
    'fmt' {
        Uvr @('ruff', 'format', 'src', 'tests', 'scripts')
        Uvr @('ruff', 'check', '--fix', 'src', 'tests', 'scripts')
    }
    'typecheck' { Uvr @('mypy') }
    'check' {
        & $PSCommandPath lint
        & $PSCommandPath typecheck
        & $PSCommandPath test
    }

    'eval' {
        if (-not $Job) { throw 'usage: .\make.ps1 eval -Job <id>' }
        Uvr @('python', '-m', 'secai.eval.run', '--job', $Job)
    }

    'clean' {
        foreach ($p in '.pytest_cache', '.mypy_cache', '.ruff_cache', '.coverage', 'coverage.xml', 'htmlcov', 'build', 'dist') {
            if (Test-Path $p) { Remove-Item -Recurse -Force $p }
        }
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' |
            ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
    }

    default { throw "unknown target '$Target' - run .\make.ps1 help" }
}
