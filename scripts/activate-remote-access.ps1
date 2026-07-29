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
$PowerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$JupyterScript = Join-Path $Repo 'scripts\start-jupyter-remote.ps1'
$TaskName = 'Supervised-Learning-Experiments JupyterLab'
$JupyterLog = Join-Path $Repo '.tmp\remote-access\jupyter.log'

if (-not (Test-Path -LiteralPath $Tailscale)) { throw "Tailscale executable not found: $Tailscale" }
if (-not (Test-Path -LiteralPath $JupyterScript)) { throw "Jupyter launcher not found: $JupyterScript" }
if (-not (Test-Path -LiteralPath (Join-Path $Repo '.venv\Scripts\jupyter-lab.exe'))) {
    throw 'The repository .venv JupyterLab executable was not found.'
}

Write-Host 'Checking OpenSSH Server...' -ForegroundColor Cyan
$sshCapability = Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*' | Select-Object -First 1
if (-not $sshCapability -or $sshCapability.State -ne 'Installed') {
    throw 'OpenSSH Server is not installed. Install OpenSSH.Server~~~~0.0.1.0 first.'
}

Write-Host 'Starting Tailscale and OpenSSH...' -ForegroundColor Cyan
Set-Service -Name Tailscale -StartupType Automatic
Start-Service -Name Tailscale
& $Tailscale set --unattended
if ($LASTEXITCODE -ne 0) { throw "Could not enable Tailscale unattended mode (exit code $LASTEXITCODE)." }
Set-Service -Name sshd -StartupType Automatic
Start-Service -Name sshd

Write-Host 'Applying Tailscale-only firewall rules...' -ForegroundColor Cyan
$sshRule = Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP-Tailscale' -ErrorAction SilentlyContinue
if (-not $sshRule) {
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP-Tailscale' `
        -DisplayName 'OpenSSH Server (Tailscale only)' -Direction Inbound `
        -Protocol TCP -LocalPort 22 -Action Allow -Profile Any `
        -RemoteAddress 100.64.0.0/10 | Out-Null
} else { Enable-NetFirewallRule -Name 'OpenSSH-Server-In-TCP-Tailscale' }

$defaultSshRule = Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue
if ($defaultSshRule) { Disable-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' }

$jupyterRule = Get-NetFirewallRule -Name 'JupyterLab-In-TCP-Tailscale' -ErrorAction SilentlyContinue
if (-not $jupyterRule) {
    New-NetFirewallRule -Name 'JupyterLab-In-TCP-Tailscale' `
        -DisplayName 'JupyterLab (Tailscale only)' -Direction Inbound `
        -Protocol TCP -LocalPort 8888 -Action Allow -Profile Any `
        -RemoteAddress 100.64.0.0/10 | Out-Null
} else { Enable-NetFirewallRule -Name 'JupyterLab-In-TCP-Tailscale' }

New-Item -Path 'HKLM:\SOFTWARE\OpenSSH' -Force | Out-Null
New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell `
    -Value 'C:\Windows\System32\cmd.exe' -PropertyType String -Force | Out-Null

Write-Host 'Creating passwordless user-at-startup Jupyter task...' -ForegroundColor Cyan
$taskUser = "$env:USERDOMAIN\$env:USERNAME"
Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and
    $_.CommandLine -match [regex]::Escape($Repo) -and
    $_.CommandLine -match 'jupyterlab|jupyter-lab' -and
    $_.CommandLine -match '8888'
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
$taskArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$JupyterScript`""
$action = New-ScheduledTaskAction -Execute $PowerShell -Argument $taskArguments
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew
$taskPrincipal = New-ScheduledTaskPrincipal -UserId $taskUser `
    -LogonType S4U -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $taskPrincipal -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName

Write-Host 'Configuring Tailscale Serve...' -ForegroundColor Cyan
& $Tailscale serve --bg --https=443 http://127.0.0.1:8888
if ($LASTEXITCODE -ne 0) { throw "Tailscale Serve failed with exit code $LASTEXITCODE." }

Start-Sleep -Seconds 5
$statusJson = & $Tailscale status --json 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Could not read the local Tailscale status.' }
$tailscaleDnsName = [string](($statusJson | ConvertFrom-Json).Self.DNSName).TrimEnd('.')
if ([string]::IsNullOrWhiteSpace($tailscaleDnsName)) { throw 'Tailscale did not report a DNS name for this machine.' }
$taskInfo = schtasks.exe /Query /TN $TaskName /FO LIST /V 2>&1
Write-Host ''
Write-Host 'Remote access activated.' -ForegroundColor Green
Write-Host "SSH:     ssh $env:USERNAME@$tailscaleDnsName"
Write-Host "Jupyter: https://$tailscaleDnsName"
Write-Host "Task:    $TaskName ($taskUser, S4U, startup)"
Write-Host "Tailscale machine DNS name: $tailscaleDnsName"
Write-Host "Log:     $JupyterLog"
Write-Host ($taskInfo | Select-String -Pattern 'Run As User|Logon Mode|Status|Last Result')
Write-Host ''
& $Tailscale serve status
if (Test-Path -LiteralPath $JupyterLog) {
    Write-Host ''
    Write-Host 'Recent Jupyter authentication URL:' -ForegroundColor Cyan
    Get-Content -LiteralPath $JupyterLog -Tail 40 | Select-String -Pattern 'http.*token=' | ForEach-Object { $_.Line }
}
