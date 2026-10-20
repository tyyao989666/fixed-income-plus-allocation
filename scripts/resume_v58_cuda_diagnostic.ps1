$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$active = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'run_v58_rebuild_pipeline|resume_v58_rng_fixed|train_v57_multitask|gate_v57_forecasts'
})
if ($active.Count -gt 0) { throw 'A V58 pipeline process is already active.' }
$report = Join-Path $root 'output/reports/corporate_action_v58'
$marker = Join-Path $report 'finite_check_fixed_diagnostic_resume.json'
if (Test-Path -LiteralPath $marker) { throw 'Diagnostic recovery already attempted; inspect logs before another launch.' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $root "output/logs/corporate_action_v58/${stamp}_cuda_diagnostic.stdout.log"
$stderr = Join-Path $root "output/logs/corporate_action_v58/${stamp}_cuda_diagnostic.stderr.log"
Copy-Item -LiteralPath (Join-Path $report 'pipeline_status.json') -Destination (Join-Path $report "${stamp}_pre_cuda_resume_status.json")
Copy-Item -LiteralPath (Join-Path $report 'pipeline_launch.json') -Destination (Join-Path $report "${stamp}_pre_cuda_resume_launch.json")
$old = $env:CUDA_LAUNCH_BLOCKING
try {
    $env:CUDA_LAUNCH_BLOCKING = '1'
    @{ attempted_at = (Get-Date).ToString('o'); checkpoint_epoch = 214; policy = 'one resume after tested equivalent finite-check and RNG fixes; runtime adaptation separately hashed'; stdout = $stdout; stderr = $stderr } | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
    $process = Start-Process -FilePath (Join-Path $root '.venv-gpu-v55/Scripts/python.exe') -ArgumentList @('-u', '-X', 'utf8', 'scripts/resume_v58_rng_fixed.py') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    @{ pid = $process.Id; launched_at = (Get-Date).ToString('o'); stdout = $stdout; stderr = $stderr; cuda_launch_blocking = $true; resume = $true } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $report 'pipeline_launch.json') -Encoding utf8
    Write-Output "Diagnostic resume launched: PID $($process.Id)"
} finally {
    $env:CUDA_LAUNCH_BLOCKING = $old
}
