$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$out = Join-Path $root 'output/reports/task_ablation_v64'
$active = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_v64_task_ablation.py|run_v63_residual_learning.py'
})
if ($active.Count -gt 0) { throw 'A relevant training process is active' }
$marker = Join-Path $out 'supervised_launch.json'
if (Test-Path -LiteralPath $marker) { throw 'Already launched; inspect before a reviewed resume' }
$smoke = Get-Content -LiteralPath (Join-Path $out 'smoke.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (-not $smoke.passed) { throw 'Smoke tests have not passed' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logs = Join-Path $root 'output/logs/task_ablation_v64'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$stdout = Join-Path $logs "${stamp}.stdout.log"
$stderr = Join-Path $logs "${stamp}.stderr.log"
$p = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/run_v64_task_ablation.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$null = $p.Handle
@{pid=$p.Id; supervisor_pid=$PID; launched_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
$p.WaitForExit()
$code = $p.ExitCode
@{exit_code=$code; exit_code_available=($null -ne $code); exited_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'process_exit.json') -Encoding utf8
if ($code -eq 0) {
    $auditOut = Join-Path $logs "${stamp}.audit.stdout.log"
    $auditErr = Join-Path $logs "${stamp}.audit.stderr.log"
    $a = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/audit_v64_task_ablation.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $auditOut -RedirectStandardError $auditErr -PassThru
    $null = $a.Handle
    $a.WaitForExit()
    @{exit_code=$a.ExitCode; stdout=$auditOut; stderr=$auditErr; exited_at=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $out 'audit_exit.json') -Encoding utf8
}
