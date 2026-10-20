$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $root '.venv-gpu-v55\Scripts\python.exe'
$report = Join-Path $root 'output\reports\total_view_rank_v56'
foreach ($arm in @('unified_v53\full', 'temporal_opportunity_v55')) {
    foreach ($year in 2022..2026) {
        $path = Join-Path $root "output\models\$arm\$year\report.json"
        if (!(Test-Path -LiteralPath $path)) { throw "Control incomplete: $path" }
        $annual = Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
        if (($annual.completed_seeds -join ',') -ne '17,29,47') { throw 'Control seeds incomplete' }
    }
}
foreach ($relative in @('unified_v53\full_training_launch.json', 'temporal_opportunity_v55\GPU_training_launch.json')) {
    $prior = Get-Content -LiteralPath (Join-Path $root "output\reports\$relative") -Raw -Encoding utf8 | ConvertFrom-Json
    if (Get-Process -Id $prior.pid -ErrorAction SilentlyContinue) { throw 'Prior control process still active' }
}
$statePath = Join-Path $report 'training_status.json'
if (Test-Path -LiteralPath $statePath) {
    $state = Get-Content -LiteralPath $statePath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($state.status -eq 'completed_objective_research') { throw 'V56 already completed; do not rerun' }
}
$launch = Join-Path $report 'GPU_training_launch.json'
$attempt = 1
if (Test-Path -LiteralPath $launch) {
    $old = Get-Content -LiteralPath $launch -Raw -Encoding utf8 | ConvertFrom-Json
    if (Get-Process -Id $old.pid -ErrorAction SilentlyContinue) { throw 'V56 process already active' }
    $attempt = [int]$old.attempt + 1
    if ($attempt -gt 2) { throw 'V56 single-resume budget exhausted' }
}
$logDir = Join-Path $root 'output\logs\total_view_rank_v56'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $report | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir ($stamp + '_gpu.stdout.log')
$stderr = Join-Path $logDir ($stamp + '_gpu.stderr.log')
$script = Join-Path $root 'scripts\run_v56_total_view_ranking.py'
$process = Start-Process -FilePath $python -ArgumentList @('-u','-X','utf8',$script,'--train','--resume') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{pid=$process.Id; attempt=$attempt; started_at=[DateTimeOffset]::Now.ToString('o'); device='cuda'; stdout=$stdout; stderr=$stderr; formal_promotion=$false; ppo_started=$false}
$record | ConvertTo-Json | Set-Content -LiteralPath $launch -Encoding utf8
Start-Sleep -Seconds 3
if ($process.HasExited) { throw "V56 exited early; inspect $stderr" }
$record | ConvertTo-Json
