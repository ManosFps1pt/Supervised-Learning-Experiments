$ErrorActionPreference = 'Stop'
$script = Join-Path $PSScriptRoot 'activate-remote-access.ps1'
Write-Host 'setup-remote-access.ps1 is retained as a compatibility alias.' -ForegroundColor Yellow
& $script
