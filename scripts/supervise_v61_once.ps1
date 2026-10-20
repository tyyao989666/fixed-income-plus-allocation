$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$report = Join-Path $root 'output/reports/state_momentum_v61'
New-Item -ItemType Directory -Force -Path $report | Out-Null
$active = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_v61_state_ranker.py|run_v60_supervised_adapter.py|run_v59_raw_mean.py'
})
if ($active.Count -gt 0) { throw 'Relevant training already active' }
$marker = Join-Path $report 'supervised_launch.json'
if (Test-Path -LiteralPath $marker) { throw 'Already launched; inspect before a reviewed resume' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logdir = Join-Path $root 'output/logs/state_momentum_v61'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$stdout = Join-Path $logdir "${stamp}.stdout.log"
$stderr = Join-Path $logdir "${stamp}.stderr.log"
$env:CUDA_LAUNCH_BLOCKING = '1'
$process = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/run_v61_state_ranker.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$null = $process.Handle
@{pid=$process.Id; supervisor_pid=$PID; launched_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
$process.WaitForExit()
@{exit_code=$process.ExitCode; exit_code_available=($null -ne $process.ExitCode); exited_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'process_exit.json') -Encoding utf8
