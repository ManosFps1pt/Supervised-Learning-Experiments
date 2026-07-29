$ErrorActionPreference = 'Stop'
$Repo = 'D:\projects\Supervised-Learning-Experiments'
$Python = Join-Path $Repo '.venv\Scripts\python.exe'
$JupyterLab = Join-Path $Repo '.venv\Scripts\jupyter-lab.exe'
$LogDir = Join-Path $Repo '.tmp\remote-access'
$Log = Join-Path $LogDir 'jupyter.log'
$RuntimeDir = Join-Path $LogDir 'runtime'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repo Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $JupyterLab)) {
    throw "JupyterLab not found in the repo environment: $JupyterLab"
}

Set-Location -LiteralPath $Repo
$env:JUPYTER_RUNTIME_DIR = $RuntimeDir
Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] Starting JupyterLab as $([Security.Principal.WindowsIdentity]::GetCurrent().Name)"
Add-Content -LiteralPath $Log -Value "Python: $Python"
Add-Content -LiteralPath $Log -Value "JupyterLab wrapper: $JupyterLab"
Add-Content -LiteralPath $Log -Value "Runtime directory: $RuntimeDir"
try {
    # Jupyter writes normal startup messages to stderr. Do not let PowerShell
    # interpret those native stderr messages as terminating script errors.
    $ErrorActionPreference = 'Continue'
    & $Python -m jupyterlab `
        --no-browser `
        --ip=0.0.0.0 `
        --port=8888 `
        --ServerApp.root_dir="$Repo" `
        --ServerApp.allow_remote_access=True `
        --ServerApp.open_browser=False `
        *>> $Log
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] JupyterLab exited with code $exitCode"
    exit $exitCode
} catch {
    Add-Content -LiteralPath $Log -Value "[$(Get-Date -Format o)] Launcher error: $($_.Exception.Message)"
    exit 1
}
