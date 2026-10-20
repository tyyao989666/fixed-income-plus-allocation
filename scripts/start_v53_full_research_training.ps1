$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = 'C:\Users\lenovo\AppData\Local\Programs\Python\Python312\python.exe'
$report = Join-Path $root 'output\reports\unified_v53'
$audit = Get-Content -LiteralPath (Join-Path $report 'data_audit.json') -Raw -Encoding utf8 | ConvertFrom-Json
$state = Get-Content -LiteralPath (Join-Path $report 'status.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (!$audit.passed -or $audit.dataset_sha256 -ne $state.dataset_sha256) {
    throw 'Current dataset has not passed its hash-bound source audit'
}
$preflight = Get-Content -LiteralPath (Join-Path $report 'forecast\training_status.json') -Raw -Encoding utf8 | ConvertFrom-Json
if ($preflight.dataset_sha256 -ne $state.dataset_sha256) {
    throw 'All-year current-dataset preflight has not passed'
}
$foldAudit = Get-Content -LiteralPath (Join-Path $report 'forecast\v45_forecast_preflight.json') -Raw -Encoding utf8 | ConvertFrom-Json
if (@($foldAudit.folds).Count -ne 5 -or @($foldAudit.folds | Where-Object { !$_.structural_selection_available }).Count -gt 0) {
    throw 'At least one annual selection contract is structurally unavailable'
}
$launchFile = Join-Path $report 'full_training_launch.json'
if (Test-Path -LiteralPath $launchFile) {
    $previous = Get-Content -LiteralPath $launchFile -Raw -Encoding utf8 | ConvertFrom-Json
    if (Get-Process -Id $previous.pid -ErrorAction SilentlyContinue) {
        throw 'Owned V53 long training is already active; refusing duplicate'
    }
}
$logDir = Join-Path $root 'output\logs\unified_v53'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir ($stamp + '_full.stdout.log')
$stderr = Join-Path $logDir ($stamp + '_full.stderr.log')
$script = Join-Path $root 'scripts\run_v53_research_training.py'
$process = Start-Process -FilePath $python -ArgumentList @('-u', '-X', 'utf8', $script, '--train', '--resume') -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$record = [ordered]@{
    pid = $process.Id
    started_at = [DateTimeOffset]::Now.ToString('o')
    stage = 'full_2022_2026_three_seed_forecast_research'
    dataset_sha256 = $state.dataset_sha256
    stdout = $stdout
    stderr = $stderr
    formal_portfolio_promoted = $false
    ppo_started = $false
    resumed_from_pid = $(if ($previous) { $previous.pid } else { $null })
    command = @($python, '-u', '-X', 'utf8', $script, '--train', '--resume')
}
$record | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $launchFile -Encoding utf8
Start-Sleep -Seconds 3
if ($process.HasExited) { throw "Training process exited early; inspect $stderr" }
$record | ConvertTo-Json -Depth 5
