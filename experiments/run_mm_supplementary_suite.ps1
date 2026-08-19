$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = "python"
$seeds = @(42, 123, 456, 789)
$configs = @(
    "experiments/configs/mm_deep_multi_real_news_2026_pw_baseline.json",
    "experiments/configs/mm_deep_multi_real_news_2026_cgcma_ablation.json",
    "experiments/configs/mm_deep_multi_real_news_2026_task_specific.json",
    "experiments/configs/mm_deep_btc_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_eth_real_news_asset_consistency.json",
    "experiments/configs/mm_deep_sol_real_news_asset_consistency.json"
)

$logPath = Join-Path $repoRoot ("experiments/results_mm/supplementary_suite_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))
"[$(Get-Date -Format s)] Starting supplementary suite" | Tee-Object -FilePath $logPath -Append

foreach ($config in $configs) {
    $configName = [System.IO.Path]::GetFileNameWithoutExtension($config)
    foreach ($seed in $seeds) {
        $existing = Get-ChildItem "experiments/results_mm" -Directory -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -like "${configName}_*_seed${seed}" -and
                (Test-Path (Join-Path $_.FullName "summary.csv"))
            } |
            Select-Object -First 1
        if ($existing) {
            "[$(Get-Date -Format s)] Skip existing $configName seed $seed -> $($existing.Name)" | Tee-Object -FilePath $logPath -Append
            continue
        }

        "[$(Get-Date -Format s)] Running $configName seed $seed" | Tee-Object -FilePath $logPath -Append
        & $python -m experiments.run_mm_deep_experiment --config $config --seed $seed 2>&1 |
            Tee-Object -FilePath $logPath -Append
        "[$(Get-Date -Format s)] Finished $configName seed $seed" | Tee-Object -FilePath $logPath -Append
    }
}

"[$(Get-Date -Format s)] Supplementary suite complete" | Tee-Object -FilePath $logPath -Append
