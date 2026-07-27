[CmdletBinding()]
param(
    [Parameter()]
    [string[]] $Task = @("all"),

    [Parameter()]
    [switch] $Preflight,

    [Parameter()]
    [switch] $Smoke
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Preflight -and $Smoke) {
    throw "-Preflight and -Smoke are mutually exclusive. Preflight does not download task assets."
}

$PortableRoot = $PSScriptRoot
$RepoRoot = (Resolve-Path (Join-Path $PortableRoot "..\..")).Path
$VenvRoot = Join-Path $RepoRoot ".venv-ioai-portable"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$Requirements = Join-Path $PortableRoot "requirements.txt"
$Bootstrap = Join-Path $PortableRoot "bootstrap.py"
$SetupTemp = Join-Path $RepoRoot ".tmp\portable-ioai"
$EnvironmentStamp = Join-Path $VenvRoot "portable-ioai-environment.json"
$KernelRoot = Join-Path $env:APPDATA "jupyter\kernels\portable-ioai"
$KernelJson = Join-Path $KernelRoot "kernel.json"

# Keep venv/pip temporary files on the same writable drive as the repository.
# This also avoids locked-down Windows user Temp folders.
New-Item -ItemType Directory -Force -Path $SetupTemp | Out-Null
$env:TEMP = $SetupTemp
$env:TMP = $SetupTemp

$SelectedTasks = @(
    foreach ($TaskValue in $Task) {
        foreach ($TaskName in ($TaskValue -split ",")) {
            $NormalizedTask = $TaskName.Trim()
            if ($NormalizedTask) {
                $NormalizedTask
            }
        }
    }
)

if ($SelectedTasks.Count -eq 0) {
    $SelectedTasks = @("all")
}

function Assert-LastExitCode {
    param(
        [Parameter(Mandatory)]
        [string] $Description
    )

    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

function New-PortableEnvironment {
    if (Test-Path -LiteralPath $VenvPython) {
        & $VenvPython -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)"
        Assert-LastExitCode "Checking the existing portable Python environment"
        & $VenvPython -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pip') else 1)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Repairing pip in the existing portable environment"
            & $VenvPython -m ensurepip --upgrade --default-pip
            Assert-LastExitCode "Repairing pip in the portable Python environment"
        }
        return
    }

    $PythonCommand = $null
    $PythonPrefixArguments = @()

    $PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        & $PyLauncher.Source -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $PythonCommand = $PyLauncher.Source
            $PythonPrefixArguments = @("-3.12")
        }
    }

    if ($null -eq $PythonCommand) {
        $PythonFallback = Get-Command "python" -ErrorAction SilentlyContinue
        if ($null -ne $PythonFallback) {
            & $PythonFallback.Source -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" *> $null
            if ($LASTEXITCODE -eq 0) {
                $PythonCommand = $PythonFallback.Source
            }
        }
    }

    if ($null -eq $PythonCommand) {
        throw "Python 3.12 was not found. Install it from python.org with the 'py' launcher, then rerun this command."
    }

    Write-Host "Creating Python 3.12 environment at $VenvRoot"
    & $PythonCommand @PythonPrefixArguments -m venv $VenvRoot
    Assert-LastExitCode "Creating the portable Python environment"
}

function Invoke-Bootstrap {
    param(
        [Parameter(Mandatory)]
        [ValidateSet("preflight", "fetch", "notebook-smoke")]
        [string] $Action
    )

    $BootstrapArguments = @($Bootstrap, $Action)
    foreach ($TaskName in $SelectedTasks) {
        $BootstrapArguments += @("--task", $TaskName)
    }

    Write-Host "Running portable IOAI $Action for: $($SelectedTasks -join ', ')"
    & $VenvPython @BootstrapArguments
    Assert-LastExitCode "Portable IOAI $Action"
}

function Test-PortableEnvironmentReady {
    if (-not (Test-Path -LiteralPath $EnvironmentStamp)) {
        return $false
    }
    try {
        $Stamp = Get-Content -Raw -LiteralPath $EnvironmentStamp | ConvertFrom-Json
        $RequirementsHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Requirements).Hash
        if ($Stamp.requirements_sha256 -ne $RequirementsHash) {
            return $false
        }
        & $VenvPython -c "import torch, torchvision, transformers, datasets, nbclient, sentence_transformers; raise SystemExit(0 if torch.__version__.startswith('2.7.1') and torchvision.__version__.startswith('0.22.1') else 1)"
        return $LASTEXITCODE -eq 0
    }
    catch {
        return $false
    }
}

function Write-PortableEnvironmentStamp {
    $Stamp = [ordered]@{
        python = "3.12"
        torch = "2.7.1+cpu"
        torchvision = "0.22.1+cpu"
        requirements_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Requirements).Hash
    }
    $Stamp | ConvertTo-Json | Set-Content -Encoding UTF8 -LiteralPath $EnvironmentStamp
}

function Register-PortableKernel {
    $KernelReady = $false
    if (Test-Path -LiteralPath $KernelJson) {
        try {
            $Kernel = Get-Content -Raw -LiteralPath $KernelJson | ConvertFrom-Json
            $KernelReady = (
                $Kernel.argv.Count -gt 0 -and
                [System.StringComparer]::OrdinalIgnoreCase.Equals(
                    [System.IO.Path]::GetFullPath([string] $Kernel.argv[0]),
                    [System.IO.Path]::GetFullPath($VenvPython)
                )
            )
        }
        catch {
            $KernelReady = $false
        }
    }
    if ($KernelReady) {
        Write-Host "The portable-ioai Jupyter kernel is already registered"
        return
    }
    Write-Host "Registering the portable-ioai Jupyter kernel"
    & $VenvPython -m ipykernel install `
        --user `
        --name "portable-ioai" `
        --display-name "Python 3.12 (portable-ioai)"
    Assert-LastExitCode "Registering the Jupyter kernel"
}

if (-not (Test-Path -LiteralPath $Requirements)) {
    throw "Missing requirements file: $Requirements"
}

if (-not (Test-Path -LiteralPath $Bootstrap)) {
    throw "Missing bootstrap utility: $Bootstrap"
}

New-PortableEnvironment

if (Test-PortableEnvironmentReady) {
    Write-Host "Pinned portable dependencies are already installed"
}
else {
    Write-Host "Installing the pinned CPU runtime"
    & $VenvPython -m pip install --disable-pip-version-check --upgrade "pip==25.1.1" "setuptools==80.9.0" "wheel==0.45.1"
    Assert-LastExitCode "Updating Python packaging tools"

    & $VenvPython -m pip install `
        --disable-pip-version-check `
        --index-url "https://download.pytorch.org/whl/cpu" `
        --extra-index-url "https://pypi.org/simple" `
        "torch==2.7.1" `
        "torchvision==0.22.1"
    Assert-LastExitCode "Installing CPU PyTorch"

    Write-Host "Installing portable notebook dependencies"
    & $VenvPython -m pip install --disable-pip-version-check --requirement $Requirements
    Assert-LastExitCode "Installing notebook dependencies"
    Write-PortableEnvironmentStamp
}

Register-PortableKernel

Invoke-Bootstrap "preflight"

if (-not $Preflight) {
    Invoke-Bootstrap "fetch"
}

if ($Smoke) {
    Invoke-Bootstrap "notebook-smoke"
}

if ($Preflight) {
    Write-Host "Preflight complete. No task datasets or model assets were downloaded."
}
elseif ($Smoke) {
    Write-Host "Portable IOAI setup and CPU notebook smoke tests completed."
}
else {
    Write-Host "Portable IOAI setup and verified asset downloads completed."
}
