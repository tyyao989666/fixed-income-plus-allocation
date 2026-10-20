$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $root '.venv-gpu-v55\Scripts\python.exe'
$report = Join-Path $root 'output\reports\temporal_opportunity_v55'
$audit = Get-Content -LiteralPath (Join-Path $report 'audit.json') -Raw -Encoding utf8 | ConvertFrom-Json
$state = Get-Content -LiteralPath (Join-Path $report 'status.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (!$audit.passed -or $audit.dataset_sha256 -ne $state.dataset_sha256) { throw 'Current V55 dataset audit missing' }
$launch = Join-Path $report 'GPU_training_launch.json'
if (Test-Path -LiteralPath $launch) {
    $old = Get-Content -LiteralPath $launch -Raw -Encoding utf8 | ConvertFrom-Json
    if (Get-Process -Id $old.pid -ErrorAction SilentlyContinue) { throw 'V55 GPU process already active' }
}
$logDir = Join-Path $root 'output\logs\temporal_opportunity_v55'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir ($stamp + '_gpu.stdout.log')
$stderr = Join-Path $logDir ($stamp + '_gpu.stderr.log')
$script = Join-Path $root 'scripts\run_v55_gpu_objective.py'
$process = Start-Process -FilePath $python -ArgumentList @('-u','-X','utf8',$script,'--train','--resume') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{pid=$process.Id; started_at=[DateTimeOffset]::Now.ToString('o'); device='cuda'; dataset_sha256=$state.dataset_sha256; stdout=$stdout; stderr=$stderr; formal_promotion=$false; ppo_started=$false}
$record | ConvertTo-Json | Set-Content -LiteralPath $launch -Encoding utf8
Start-Sleep -Seconds 3
if ($process.HasExited) { throw "CUDA challenger exited early; inspect $stderr" }
$record | ConvertTo-Json
