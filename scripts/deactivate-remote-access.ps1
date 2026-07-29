$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host 'Administrator permission is required. Opening an elevated PowerShell window...' -ForegroundColor Yellow
    Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`""
    ) -Wait
    exit
}

$Repo = 'D:\projects\Supervised-Learning-Experiments'
$Tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
$TaskName = 'Supervised-Learning-Experiments JupyterLab'

Write-Host 'Stopping and disabling Jupyter startup task...' -ForegroundColor Cyan
& schtasks.exe /End /TN $TaskName 2>$null | Out-Null
& schtasks.exe /Change /TN $TaskName /DISABLE 2>$null | Out-Null

Write-Host 'Stopping repository Jupyter processes...' -ForegroundColor Cyan
$jupyterProcesses = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($Repo) -and
    $_.CommandLine -match 'jupyter-lab|jupyter lab' -and
    $_.CommandLine -match '8888'
}
foreach ($process in $jupyterProcesses) {
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host 'Removing Tailscale Serve configuration...' -ForegroundColor Cyan
& $Tailscale serve reset

Write-Host 'Disabling remote-access firewall rules...' -ForegroundColor Cyan
Disable-NetFirewallRule -Name 'OpenSSH-Server-In-TCP-Tailscale' -ErrorAction SilentlyContinue
Disable-NetFirewallRule -Name 'JupyterLab-In-TCP-Tailscale' -ErrorAction SilentlyContinue

Write-Host 'Stopping and disabling Tailscale and OpenSSH...' -ForegroundColor Cyan
Set-Service -Name sshd -StartupType Disabled
Stop-Service -Name sshd -Force -ErrorAction SilentlyContinue
Set-Service -Name Tailscale -StartupType Disabled
Stop-Service -Name Tailscale -Force -ErrorAction SilentlyContinue

Write-Host ''
Write-Host 'Remote access deactivated.' -ForegroundColor Green
