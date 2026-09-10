# L9C train launcher (#346): l9 recipe verbatim, corpus splits (read-only),
# BelowNormal for the trampoline AND the re-exec'd child, GPU serial.
param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$Seed,
    [string]$ExtraArgs = ""
)
$ErrorActionPreference = "Stop"
$Py = "C:\Users\Julian\Projects\soundswitch-exp-ceiling-worktree\.venv\Scripts\python.exe"
$TreePB = "C:\Users\Julian\Projects\soundswitch-phase-b-worktree"
$Data = "C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform"
$Camp = "$Data\models\l9c_campaign"
$FeatDir = "$Data\features_stream\MERT-v1-330M_L6-22_F3_hop1"
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
New-Item -ItemType Directory -Force "$Camp\$RunName" | Out-Null
$log = "$Camp\$RunName\train.$stamp.log"

$pool = (Get-Counter '\Memory\Pool Nonpaged Bytes').CounterSamples[0].CookedValue / 1e9
if ($pool -ge 8.0) { Write-Output "POOL GATE TRIPPED: $pool GB >= 8 GB - refusing to launch a GPU stage"; exit 3 }
$freeMB = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024
if ($freeMB -lt 2048) { Write-Output "RAM GATE: only $freeMB MB free (< 2048) - refusing"; exit 4 }
Write-Output "preflight: pool=$([math]::Round($pool,3))GB freeRAM=$([math]::Round($freeMB))MB"

$argv = @("-u","-m","training.nn.ceiling.train_head",
    "--data-dir",$Data,"--model-dir",$Camp,"--run-name",$RunName,
    "--arm","online_crnn","--feature-dir",$FeatDir,
    "--layers","6","22","--backward-cells","41",
    "--input-affine","$Data\models\phase_b\input_affine_F3.npz",
    "--label-space","$Data\models\l9\priors.json",
    "--posteriors-dir","$Camp\posteriors_$RunName",
    "--batch-size","1","--eval-batch-size","1",
    "--crop-sec","300","--crops-per-track","3",
    "--lr","3e-4","--warmup-steps","0",
    "--epochs","20","--evals-per-epoch","1","--patience","5",
    "--seed",$Seed,"--ram-floor-gb","0.35",
    "--cache-bytes","1073741824",
    "--tb-dir","$Camp\tb","--device","cuda","--rnn-hidden","128")
if ($ExtraArgs -ne "") { $argv += ($ExtraArgs -split " ") }

$p = Start-Process -FilePath $Py -ArgumentList $argv -WorkingDirectory $TreePB `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru
try { $p.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch {}
Write-Output "launched $RunName pid=$($p.Id) log=$log"

Start-Sleep -Seconds 60
$kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($p.Id)"
foreach ($k in $kids) {
    try {
        (Get-Process -Id $k.ProcessId).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal
        Write-Output "demoted child pid=$($k.ProcessId) ($($k.Name))"
    } catch {}
}

Wait-Process -Id $p.Id
$report = "$Camp\$RunName\training_report.json"
if (Test-Path $report) {
    Write-Output "TRAIN OK: $report present"
    exit 0
} else {
    Write-Output "TRAIN FAILED: no training_report.json (see $log / $log.err)"
    exit 1
}
