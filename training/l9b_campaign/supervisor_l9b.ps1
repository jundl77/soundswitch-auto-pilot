# nextgen retrain campaign supervisor (#341). Autonomous overnight chain:
# prep -> extract -> splits -> overlay -> validate -> priors -> 4 trains ->
# 2 sweeps -> verdict -> exports -> probe sims. Stages are idempotent
# (skip-if-artifact-present); a train failure stops its arm, not the chain.
$ErrorActionPreference = "Stop"
$Repo     = "C:\Users\Julian\Projects\soundswitch-auto-pilot"
$Data     = "$Repo\training\data\raveform"
$Camp     = "$Data\models\nextgen_campaign"
$Scripts  = "$Repo\training\nextgen_campaign"
$TreePB   = "C:\Users\Julian\Projects\soundswitch-phase-b-worktree"
$PyExp    = "C:\Users\Julian\Projects\soundswitch-exp-ceiling-worktree\.venv\Scripts\python.exe"
$PyMain   = "$Repo\.venv\Scripts\python.exe"
$FeatDir  = "$Data\features_stream\MERT-v1-330M_L6-22_F3_hop1"
$Stamp    = Get-Date -Format "yyyyMMdd-HHmmss"
$StateFile = "$Camp\state.json"
$SupLog   = "$Camp\logs\supervisor.$Stamp.log"
$DiskFloorGB = 20.0
$RamGateMB   = 600       # D5 rev2: owner directive "start everything"; GPU freed by show exit; brake below covers the floor
$RamBrakeMB  = 300       # 3 consecutive minutes under this kills the child
$Overlay  = "$Camp\drop_demotion_overlay.json"

New-Item -ItemType Directory -Force "$Camp\logs" | Out-Null
(Get-Process -Id $PID).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal

$script:State = [ordered]@{
    campaign = "nextgen_campaign (#341)"; stamp = $Stamp; supervisor_pid = $PID
    status = "running"; stage = "boot"; stage_index = 0
    stages = @("preflight","prep","extract","splits","overlay","validate","priors",
               "train_ng_H_w128_s1234","train_ng_H_w128_s1235",
               "train_ng_HD_w128_s1234","train_ng_HD_w128_s1235",
               "sweep_H","sweep_HD","verdict","exports","probes","done")
    started_utc = (Get-Date).ToUniversalTime().ToString("o"); updated_utc = ""
    child_pid = $null; arm_H_failed = $false; arm_HD_failed = $false
    eta_note = "prep ~40m; trains 4x~45m (GPU contention with owner's live show may stretch this heavily); sweeps 2x~40m; verdict ~30m; probes ~60m"
    last_error = $null; notes = @()
}

function Save-State {
    $script:State.updated_utc = (Get-Date).ToUniversalTime().ToString("o")
    $tmp = "$StateFile.tmp"
    $script:State | ConvertTo-Json -Depth 6 | Out-File -FilePath $tmp -Encoding utf8
    Move-Item -Force $tmp $StateFile
}
function Note([string]$line) {
    $entry = "$(Get-Date -Format o)  $line"
    $script:State.notes += $entry
    $entry | Out-File -FilePath $SupLog -Encoding utf8 -Append
    Save-State
}
function Fail-Chain([string]$why) {
    $script:State.status = "failed"; $script:State.last_error = $why
    Note "FAILED: $why"
    exit 1
}
function Set-Stage([string]$name, [int]$index) {
    $script:State.stage = $name; $script:State.stage_index = $index
    Note "stage -> $name"
}
function Get-AvailMB { [math]::Round((Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue) }
function Wait-ForMemory([string]$what) {
    $deadline = (Get-Date).AddHours(12); $lastNote = Get-Date "2000-01-01"
    while ($true) {
        $avail = Get-AvailMB
        if ($avail -ge $RamGateMB) { return }
        if ((Get-Date) -gt $deadline) { Fail-Chain "$what : waited 12h for $RamGateMB MB available (last $avail)" }
        if (((Get-Date) - $lastNote).TotalMinutes -ge 15) { Note "$what : holding for RAM, $avail MB available, need $RamGateMB MB"; $lastNote = Get-Date }
        Start-Sleep -Seconds 60
    }
}
function Assert-Disk([string]$what) {
    $free = [math]::Round((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace/1GB, 1)
    if ($free -lt $DiskFloorGB) { Fail-Chain "$what : C: at $free GB, floor $DiskFloorGB" }
}

# Runs a child detached-from-shell, BelowNormal, with heartbeat, disk floor,
# RAM brake, and a stall watchdog on the log file's growth. Returns a verdict
# string: ok | exit:<code> | stalled | braked.
function Run-Child([string]$name, [string]$exe, [object[]]$argv, [string]$log, [string]$wd,
                   [int]$StallMinutes, [int]$CeilingMinutes) {
    "=== $name starting $(Get-Date -Format o) ===" | Out-File -FilePath $log -Encoding utf8
    $child = Start-Process -FilePath $exe -ArgumentList $argv -WorkingDirectory $wd `
        -NoNewWindow -PassThru -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $null = $child.Handle
    try { $child.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::BelowNormal } catch {}
    $script:State.child_pid = $child.Id; Save-State
    $start = Get-Date; $brakeStrikes = 0
    while (-not $child.HasExited) {
        Start-Sleep -Seconds 60
        Save-State
        $free = [math]::Round((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace/1GB, 1)
        if ($free -lt $DiskFloorGB) {
            try { taskkill /PID $child.Id /T /F | Out-Null } catch {}
            Fail-Chain "$name : disk floor breached mid-stage ($free GB)"
        }
        $avail = Get-AvailMB
        if ($avail -lt $RamBrakeMB) { $brakeStrikes++ } else { $brakeStrikes = 0 }
        if ($brakeStrikes -ge 3) {
            try { taskkill /PID $child.Id /T /F | Out-Null } catch {}
            $script:State.child_pid = $null
            Note "$name : RAM emergency brake ($avail MB available), child killed"
            return "braked"
        }
        $ageMin = 999
        foreach ($f in @($log, "$log.err")) {
            if (Test-Path $f) {
                $a = ((Get-Date) - (Get-Item $f).LastWriteTime).TotalMinutes
                if ($a -lt $ageMin) { $ageMin = $a }
            }
        }
        if ($StallMinutes -gt 0 -and $ageMin -gt $StallMinutes) {
            try { taskkill /PID $child.Id /T /F | Out-Null } catch {}
            $script:State.child_pid = $null
            Note "$name : stall watchdog fired (log silent $([math]::Round($ageMin)) min), child killed"
            return "stalled"
        }
        if ($CeilingMinutes -gt 0 -and ((Get-Date) - $start).TotalMinutes -gt $CeilingMinutes) {
            try { taskkill /PID $child.Id /T /F | Out-Null } catch {}
            $script:State.child_pid = $null
            Note "$name : wall ceiling $CeilingMinutes min hit, child killed"
            return "stalled"
        }
    }
    $script:State.child_pid = $null; Save-State
    $code = $child.ExitCode
    if ($null -eq $code -or "$code" -eq "") { return "ok" }   # artifact checks decide below
    if ($code -ne 0) { return "exit:$code" }
    return "ok"
}

# GPU stages get exactly one recorded retry after a stall (never a blind loop).
function Run-GpuStage([string]$name, [string]$exe, [object[]]$argv, [string]$logBase, [string]$wd,
                      [int]$StallMinutes, [int]$CeilingMinutes) {
    Wait-ForMemory $name; Assert-Disk $name
    $v = Run-Child $name $exe $argv "$logBase.$Stamp.log" $wd $StallMinutes $CeilingMinutes
    if ($v -eq "ok") { return $v }
    Note "$name : first attempt verdict '$v' -- one retry after 10 min (fresh log)"
    Start-Sleep -Seconds 600
    Wait-ForMemory "$name retry"
    $v2 = Run-Child "$name retry" $exe $argv "$logBase.$Stamp.retry.log" $wd $StallMinutes $CeilingMinutes
    return $v2
}

# ---------- preflight ----------
Set-Stage "preflight" 0
foreach ($f in @("$Scripts\ng_prep.py", "$Scripts\ng_arm_splits.py", "$Scripts\ng_build_overlay.py",
                 "$Scripts\ng_priors_refit.py",
                 "$Data\models\spec_mapped_campaign\mellow_drop_measurement\drop_sections.csv",
                 "$TreePB\training\nn\ceiling\validate_arm_datasets.py")) {
    if (-not (Test-Path $f)) { Fail-Chain "missing prerequisite: $f" }
}
# Decode-side scripts land while the early stages run (#316 pattern): their
# stages park until the file appears rather than failing a launchable chain.
function Wait-ForScript([string]$path, [string]$what) {
    $deadline = (Get-Date).AddHours(3)
    while (-not (Test-Path $path)) {
        if ((Get-Date) -gt $deadline) { return $false }
        Note "$what : waiting for $path to land"
        Start-Sleep -Seconds 120
        Save-State
    }
    return $true
}
$branch = (git -C $TreePB rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "nextgen_trainer") { Fail-Chain "phase-b worktree on '$branch', need nextgen_trainer" }
Assert-Disk "preflight"
Note "preflight ok: scripts present, phase-b on nextgen_trainer @ $((git -C $TreePB rev-parse --short HEAD).Trim()), avail RAM $(Get-AvailMB) MB"

# ---------- prep (admission verify + extraction inputs) ----------
Set-Stage "prep" 1
$v = Run-Child "prep" $PyMain @("-u", "$Scripts\ng_prep.py") "$Camp\logs\prep.$Stamp.log" $Repo 30 60
if ($v -ne "ok") { Fail-Chain "prep verdict '$v' -- see logs\prep.$Stamp.log" }

# ---------- extract (GPU: F3 sidecars for the new hand tracks) ----------
Set-Stage "extract" 2
$missing = @(Get-Content "$Camp\extract_ids.txt" | Where-Object { $_ -and -not (Test-Path "$FeatDir\$_.npz") })
if ($missing.Count -eq 0) {
    Note "extract: all sidecars already present -- skipping"
} else {
    $argv = @("-u", "-m", "training.nn.ceiling.stream_extract", "extract",
              "--data-dir", $Data, "--margins", "3", "--hop-sec", "1",
              "--ids-file", "$Camp\extract_ids.txt", "--audio-map", "$Camp\audio_map.json",
              "--report", "$Camp\stream_extract.nextgen.$Stamp.json")
    $v = Run-GpuStage "extract" $PyExp $argv "$Camp\logs\extract" $TreePB 45 240
    if ($v -ne "ok") { Fail-Chain "extract verdict '$v' after retry -- everything downstream needs the sidecars" }
}
$v = Run-Child "post-extract check" $PyMain @("-u", "$Scripts\ng_prep.py", "--post-extract") "$Camp\logs\prep_post.$Stamp.log" $Repo 15 30
if ($v -ne "ok") { Fail-Chain "post-extract verification failed" }

# ---------- splits arming ----------
Set-Stage "splits" 3
$v = Run-Child "splits" $PyMain @("-u", "$Scripts\ng_arm_splits.py", "--apply") "$Camp\logs\splits.$Stamp.log" $Repo 15 30
if ($v -ne "ok") { Fail-Chain "splits arming verdict '$v' (script restores backup on assertion failure)" }

# ---------- overlay ----------
Set-Stage "overlay" 4
$v = Run-Child "overlay" $PyMain @("-u", "$Scripts\ng_build_overlay.py", "--splits-file", "$Data\splits.json") "$Camp\logs\overlay.$Stamp.log" $Repo 15 30
if ($v -ne "ok") { Fail-Chain "overlay build verdict '$v'" }

# ---------- validate arm datasets ----------
Set-Stage "validate" 5
$checks = @("--check-hand-id", "xSSy3aHsG6c", "--check-hand-id", "hand-65cb8c94812d", "--check-hand-id", "hand-8339586c555a")
$v = Run-Child "validate H" $PyExp (@("-u", "-m", "training.nn.ceiling.validate_arm_datasets", "--data-dir", $Data) + $checks) "$Camp\logs\validate_H.$Stamp.log" $TreePB 15 30
if ($v -ne "ok") { Fail-Chain "arm H dataset validation failed" }
$v = Run-Child "validate HD" $PyExp (@("-u", "-m", "training.nn.ceiling.validate_arm_datasets", "--data-dir", $Data, "--demote-drops", $Overlay) + $checks) "$Camp\logs\validate_HD.$Stamp.log" $TreePB 15 30
if ($v -ne "ok") { Fail-Chain "arm HD dataset validation failed" }

# ---------- priors (post-arming, per arm) ----------
Set-Stage "priors" 6
$v = Run-Child "priors H" $PyMain @("-u", "$Scripts\ng_priors_refit.py", "--arm", "H", "--out", "$Camp\priors_H.json") "$Camp\logs\priors_H.$Stamp.log" $Repo 20 60
if ($v -ne "ok") { Fail-Chain "priors H verdict '$v'" }
$v = Run-Child "priors HD" $PyMain @("-u", "$Scripts\ng_priors_refit.py", "--arm", "HD", "--overlay", $Overlay, "--out", "$Camp\priors_HD.json") "$Camp\logs\priors_HD.$Stamp.log" $Repo 20 60
if ($v -ne "ok") { Fail-Chain "priors HD verdict '$v'" }

# ---------- trains (l9 recipe verbatim; HD adds --demote-drops) ----------
function Train-Args([string]$run, [string]$seed, [bool]$demote) {
    $argv = @("-u", "-m", "training.nn.ceiling.train_head",
        "--data-dir", $Data, "--model-dir", $Camp, "--run-name", $run,
        "--arm", "online_crnn", "--feature-dir", $FeatDir,
        "--layers", "6", "22", "--backward-cells", "41",
        "--input-affine", "$Data\models\phase_b\input_affine_F3.npz",
        "--label-space", "$Data\models\l9\priors.json",
        "--posteriors-dir", "$Camp\posteriors_$run",
        "--batch-size", "1", "--eval-batch-size", "1",
        "--crop-sec", "300", "--crops-per-track", "3",
        "--lr", "3e-4", "--warmup-steps", "0",
        "--epochs", "20", "--evals-per-epoch", "1", "--patience", "5",
        "--seed", $seed, "--ram-floor-gb", "0.35",
        "--cache-bytes", "1073741824",
        "--tb-dir", "$Camp\tb", "--device", "cuda", "--rnn-hidden", "128")
    if ($demote) { $argv += @("--demote-drops", $Overlay) }
    return $argv
}
$trains = @(
    @{ run = "ng_H_w128_s1234";  seed = "1234"; demote = $false; arm = "H";  index = 7 },
    @{ run = "ng_H_w128_s1235";  seed = "1235"; demote = $false; arm = "H";  index = 8 },
    @{ run = "ng_HD_w128_s1234"; seed = "1234"; demote = $true;  arm = "HD"; index = 9 },
    @{ run = "ng_HD_w128_s1235"; seed = "1235"; demote = $true;  arm = "HD"; index = 10 }
)
foreach ($t in $trains) {
    $run = $t.run
    Set-Stage "train_$run" $t.index
    if ($script:State["arm_$($t.arm)_failed"]) { Note "train_$run : arm $($t.arm) already failed -- skipping"; continue }
    $report = "$Camp\$run\training_report.json"
    if (Test-Path $report) { Note "train_$run : already complete -- skipping"; continue }
    New-Item -ItemType Directory -Force "$Camp\$run" | Out-Null
    $v = Run-GpuStage "train_$run" $PyExp (Train-Args $run $t.seed $t.demote) "$Camp\$run\train" $TreePB 60 480
    $finished = Test-Path $report
    if (-not $finished) {
        $script:State["arm_$($t.arm)_failed"] = $true
        Note "train_$run FAILED (verdict '$v', no training_report.json) -- arm $($t.arm) stopped, continuing other arm"
    } else {
        Note "train_$run complete (verdict '$v', report present)"
    }
}
if ($script:State.arm_H_failed -and $script:State.arm_HD_failed) { Fail-Chain "both arms failed in training" }

# ---------- sweeps ----------
foreach ($arm in @("H", "HD")) {
    Set-Stage "sweep_$arm" $(if ($arm -eq "H") { 11 } else { 12 })
    if ($script:State["arm_${arm}_failed"]) { Note "sweep_$arm : arm failed -- skipping"; continue }
    if (Test-Path "$Camp\decoder_config_$arm.json") { Note "sweep_$arm : config already present -- skipping"; continue }
    if (-not (Wait-ForScript "$Scripts\ng_sweep_driver.py" "sweep_$arm")) { $script:State["arm_${arm}_failed"] = $true; Note "sweep_$arm : driver never landed"; continue }
    Wait-ForMemory "sweep_$arm"
    $v = Run-Child "sweep_$arm" $PyMain @("-u", "$Scripts\ng_sweep_driver.py", "--arm", $arm) "$Camp\logs\sweep_$arm.$Stamp.log" $Repo 60 360
    if ($v -ne "ok" -or -not (Test-Path "$Camp\decoder_config_$arm.json")) {
        $script:State["arm_${arm}_failed"] = $true
        Note "sweep_$arm FAILED (verdict '$v') -- arm $arm stopped"
    }
}
if ($script:State.arm_H_failed -and $script:State.arm_HD_failed) { Fail-Chain "both arms failed by the sweep stage" }

# ---------- verdict ----------
Set-Stage "verdict" 13
if (-not (Wait-ForScript "$Scripts\ng_decoded_verdict.py" "verdict")) { Fail-Chain "verdict script never landed" }
Wait-ForMemory "verdict"
$v = Run-Child "verdict" $PyMain @("-u", "$Scripts\ng_decoded_verdict.py", "--allow-missing-arms") "$Camp\logs\verdict.$Stamp.log" $Repo 45 180
if ($v -ne "ok") { Note "verdict verdict '$v' -- recorded, chain continues (probes are independent)" }

# ---------- exports + shadow chains ----------
Set-Stage "exports" 14
if (-not (Wait-ForScript "$Scripts\ng_probe_rig.py" "exports")) { Fail-Chain "probe rig never landed" }
$armRuns = @()
foreach ($t in $trains) { if (-not $script:State["arm_$($t.arm)_failed"]) { $armRuns += @{ run = $t.run; arm = $t.arm } } }
foreach ($r in $armRuns) {
    $v = Run-Child "export $($r.run)" $PyMain @("-u", "$Scripts\ng_probe_rig.py", "export-arm", "--arm", $r.arm, "--run", $r.run) "$Camp\logs\export_$($r.run).$Stamp.log" $Repo 30 90
    if ($v -ne "ok") { Note "export $($r.run) FAILED (verdict '$v') -- its probe rows will be missing" }
}

# ---------- probes (2 tracks x shipped + each surviving run) ----------
Set-Stage "probes" 15
$tracks = @("hand-65cb8c94812d", "hand-8339586c555a")
foreach ($track in $tracks) {
    $v = Run-GpuStage "probe shipped $track" $PyMain @("-u", "$Scripts\ng_probe_rig.py", "run", "--chain", "shipped", "--track", $track) "$Camp\logs\probe_shipped_$track" $Repo 45 180
    if ($v -ne "ok") { Note "probe shipped/$track verdict '$v' -- recorded" }
    foreach ($r in $armRuns) {
        $v = Run-Child "probe $($r.run) $track" $PyMain @("-u", "$Scripts\ng_probe_rig.py", "run", "--chain", $r.run, "--track", $track) "$Camp\logs\probe_$($r.run)_$track.$Stamp.log" $Repo 45 180
        if ($v -ne "ok") { Note "probe $($r.run)/$track verdict '$v' -- recorded" }
    }
}
$v = Run-Child "probe collate" $PyMain @("-u", "$Scripts\ng_probe_rig.py", "collate") "$Camp\logs\probe_collate.$Stamp.log" $Repo 15 30
if ($v -ne "ok") { Note "probe collate verdict '$v'" }

Set-Stage "done" 16
$script:State.status = "done"
Note "chain complete: verdict at $Camp\NG_DECODED.json, probes at $Camp\PROBES.md; RESULTS.md + decisions-log addenda are the coordinator's morning step"
Save-State
