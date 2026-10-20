$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$report = Join-Path $root 'output/reports/corporate_action_v58'
Start-Transcript -Path (Join-Path $report 'supervisor_startup.log') -Append
$active = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_v58_rebuild_pipeline|resume_v58_rng_fixed|train_v57_multitask|gate_v57_forecasts'
})
if ($active.Count -gt 0) { throw 'V58 already active' }
$marker = Join-Path $report 'supervised_resume_epoch247.json'
if (Test-Path -LiteralPath $marker) { throw 'This checkpoint recovery has already been launched; inspect before retrying' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $root "output/logs/corporate_action_v58/${stamp}_supervised.stdout.log"
$stderr = Join-Path $root "output/logs/corporate_action_v58/${stamp}_supervised.stderr.log"
foreach ($name in @('pipeline_status.json', 'pipeline_launch.json', 'rng_resume_provenance.json')) {
    Copy-Item -LiteralPath (Join-Path $report $name) -Destination (Join-Path $report "${stamp}_previous_$name")
}
@{started_at=(Get-Date).ToString('o'); supervisor_pid=$PID; checkpoint_epoch=247; auto_restart=$false} | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
$env:CUDA_LAUNCH_BLOCKING = '1'
$process = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u','-X','utf8','scripts/resume_v58_rng_fixed.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$null = $process.Handle
@{pid=$process.Id; supervisor_pid=$PID; launched_at=(Get-Date).ToString('o'); stdout=$stdout; stderr=$stderr; resume=$true; cuda_launch_blocking=$true; launch_method='hidden_single_run_supervisor'} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'pipeline_launch.json') -Encoding utf8
$process.WaitForExit()
@{exit_code=$process.ExitCode; exit_code_available=($null -ne $process.ExitCode); exited_at=(Get-Date).ToString('o'); pid=$process.Id; supervisor_pid=$PID; auto_restart=$false; stdout=$stdout; stderr=$stderr} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report "${stamp}_process_exit.json") -Encoding utf8
Stop-Transcript
