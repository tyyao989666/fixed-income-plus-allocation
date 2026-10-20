$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$report = Join-Path $root 'output/reports/residual_learning_v63'
$active = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_v63_residual_learning.py|run_v61_state_ranker.py'
})
if ($active.Count -gt 0) { throw 'Relevant training already active' }
$marker = Join-Path $report 'supervised_launch.json'
if (Test-Path -LiteralPath $marker) { throw 'Already launched; review before any manual resume' }
$smoke = Get-Content -LiteralPath (Join-Path $report 'smoke.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $smoke.passed) { throw 'GPU smoke test not passed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logdir = Join-Path $root 'output/logs/residual_learning_v63'
New-Item -ItemType Directory -Force -Path $logdir | Out-Null
$stdout = Join-Path $logdir "${stamp}.stdout.log"
$stderr = Join-Path $logdir "${stamp}.stderr.log"
$process = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/run_v63_residual_learning.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$null = $process.Handle
@{pid=$process.Id; supervisor_pid=$PID; launched_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
$process.WaitForExit()
$code = $process.ExitCode
@{exit_code=$code; exit_code_available=($null -ne $code); exited_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'process_exit.json') -Encoding utf8
if ($code -eq 0) {
    $auditOut = Join-Path $logdir "${stamp}.audit.stdout.log"
    $auditErr = Join-Path $logdir "${stamp}.audit.stderr.log"
    $audit = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/audit_v63_results.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $auditOut -RedirectStandardError $auditErr -PassThru
    $null = $audit.Handle
    $audit.WaitForExit()
    @{exit_code=$audit.ExitCode; stdout=$auditOut; stderr=$auditErr; exited_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'audit_exit.json') -Encoding utf8
}
