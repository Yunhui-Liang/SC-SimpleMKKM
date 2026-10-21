$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

python -m scsimplemkkm.cli run-mv --config configs/mfeat_rbf_self_tuned_seed0.json
