# Raveform Data Acquisition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the Raveform dataset onto this machine — all 1,423 expert EDM section annotations plus every track's audio that is still available on YouTube — as the training corpus for the classifier upgrade.

**Architecture:** Annotations come from the Hugging Face dataset `taejunkim/raveform` (annotations + YouTube IDs only; no audio is hosted). Audio is fetched per-YouTube-ID with yt-dlp → 192k mp3, into a gitignored data directory. A committed manifest builder and downloader script make the whole corpus reproducible; the bulk download itself runs as a detached OS process that survives this session. Everything must be resumable — expect some fraction of the 1,423 videos to be dead or region-blocked; record failures, don't fight them.

**Tech Stack:** Python 3.12 (repo venv via uv), yt-dlp, ffmpeg, Hugging Face raw-file HTTP endpoints (no HF token needed for public datasets).

## Global Constraints

- Branch: `raveform_data_pipeline` (create off `master`). Never commit to `master`. You work in a git worktree — **all downloaded DATA must go to the main repository's absolute path** `C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\`, NOT into the worktree (the worktree is temporary; the data must survive it). Code/scripts are committed in the worktree on the branch.
- `training/data/` is gitignored (Task 1) — audio and annotations never enter git.
- Be polite to YouTube: sequential downloads, sleep between videos, bounded retries. If yt-dlp reports bot-check / sign-in-required errors, do NOT use `--cookies-from-browser` or any credential workaround — record the failure and report it; the owner decides.
- Personal research use on the owner's machine, standard MIR practice for YouTube-ID datasets (AudioSet-style). Download audio only (`-x`), never video.
- All shell examples below are PowerShell-compatible; the scripts themselves must be pure Python (subprocess → yt-dlp) so they run identically from any shell.

---

### Task 1: Tooling + gitignore

**Files:**
- Modify: `.gitignore`
- Create: none

- [ ] **Step 1: Gitignore the data tree**

Append to `.gitignore`:

```
training/data/
```

(`training/data/` already holds untracked SALAMI/DEAM material from 2025 experiments; this makes the ignore explicit before gigabytes of audio land there.)

- [ ] **Step 2: Verify/install ffmpeg**

Run: `ffmpeg -version`
If missing: `winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements`, then open a fresh shell (PATH refresh) and re-verify. If winget is unavailable, report back instead of improvising an installer.

- [ ] **Step 3: Verify/install yt-dlp**

Run: `yt-dlp --version`
If missing: `uv tool install yt-dlp`, re-verify (`uv tool` shims are on PATH). Record the installed version in your final report — yt-dlp behavior is version-sensitive.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "gitignore training/data — raveform corpus lands there"
```

---

### Task 2: Download and validate the annotations

**Files:**
- Create: `training/raveform_fetch_annotations.py`
- Data (main repo absolute path): `C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\annotations\`

**Interfaces:**
- Produces: annotation files on disk + a printed validation summary. Task 3 consumes the annotation directory layout this task discovers.

- [ ] **Step 1: Discover the dataset layout**

List the repo files: `https://huggingface.co/api/datasets/taejunkim/raveform/tree/main` (GET, JSON; recurse into subfolders via `.../tree/main/<subpath>`). Identify the annotation archive or per-track annotation files and any metadata/index file mapping track IDs → YouTube IDs. Raw file download pattern: `https://huggingface.co/datasets/taejunkim/raveform/resolve/main/<path>`.

- [ ] **Step 2: Write the fetch script**

`training/raveform_fetch_annotations.py`: stdlib-only (`urllib.request`, `zipfile`, `json`, `argparse`), with `--data-dir` defaulting to `<repo-root>/training/data/raveform` (derive repo root from `Path(__file__).resolve().parents[1]`) — **when you run it from the worktree, pass `--data-dir C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform` explicitly.** It downloads the annotation archive(s) to `<data-dir>/annotations/`, extracts if zipped, and is idempotent (skips files that already exist with non-zero size).

Adapt the internals to the layout found in Step 1 — the script's contract, not its internals, is fixed: after running, `<data-dir>/annotations/` contains all annotation files, and the script prints how many tracks it found.

- [ ] **Step 3: Run it and validate**

Run it, then validate against the published facts and print a summary:
- Track count ≈ 1,423 (print the exact number found).
- Parse 3 sample annotation files end-to-end; print one parsed example (list of `(start_sec, end_sec, label)`).
- Collect the label vocabulary across ALL files; it must be within {intro, buildup, breakdown, drop, cooldown, outro, altoutro} (case/formatting may differ — print the raw set you actually observed).
- Every track must yield a YouTube ID; print 3 examples and the count of tracks with IDs.

If any of these deviates materially (different labels, missing IDs), STOP and report before continuing — Task 3+ depend on the schema.

- [ ] **Step 4: Commit**

```bash
git add training/raveform_fetch_annotations.py
git commit -m "raveform: annotation fetcher + schema validation"
```

---

### Task 3: Manifest + label statistics

**Files:**
- Create: `training/raveform_manifest.py`
- Data: `<data-dir>/manifest.csv`

**Interfaces:**
- Consumes: `<data-dir>/annotations/` from Task 2.
- Produces: `manifest.csv` with header `track_id,youtube_id,n_sections,total_sec` (one row per track) — Task 4's downloader reads column `youtube_id`. Also prints label statistics.

- [ ] **Step 1: Write the manifest builder**

`training/raveform_manifest.py` (same `--data-dir` convention as Task 2): parses every annotation file, writes `manifest.csv`, and prints:
- total tracks, total annotated hours;
- per-label: section count, total duration, median section duration (these are the HSMM duration priors preview — they go in your final report);
- label transition counts (which label follows which — the transition priors preview).

- [ ] **Step 2: Run and sanity-check**

Run it. Sanity: drop should be among the most frequent labels; median drop section length should be in the tens of seconds. Paste the printed stats into your final report.

- [ ] **Step 3: Commit**

```bash
git add training/raveform_manifest.py
git commit -m "raveform: manifest + duration/transition label statistics"
```

---

### Task 4: Downloader script

**Files:**
- Create: `training/raveform_download.py`

**Interfaces:**
- Consumes: `manifest.csv` (column `youtube_id`).
- Produces: `<data-dir>/audio/<youtube_id>.mp3` files; `<data-dir>/downloaded.txt` (yt-dlp archive, resume state); `<data-dir>/failed.jsonl` (one `{"youtube_id":..., "error":...}` per failure); progress lines on stdout.

- [ ] **Step 1: Write the downloader**

`training/raveform_download.py` with `--data-dir` (same convention), `--limit N` (0 = all), `--sleep-min 2 --sleep-max 5`. Per YouTube ID not already in `downloaded.txt`, invoke:

```
yt-dlp -f bestaudio -x --audio-format mp3 --audio-quality 192K
       --no-playlist --retries 3 --socket-timeout 30
       --download-archive <data-dir>/downloaded.txt
       -o "<data-dir>/audio/%(id)s.%(ext)s"
       -- <youtube_id>
```

via `subprocess.run` (list-form args, never shell strings — IDs can start with `-`, hence the `--` separator). Non-zero exit → append to `failed.jsonl` with the tail of stderr, continue to the next ID. Sleep a fixed pseudo-random interval in `[sleep-min, sleep-max]` between downloads. Print progress every 10 tracks: `done/failed/remaining, ETA`. The script must be safe to kill and re-run at any time (`downloaded.txt` + skip-if-failed-already is the resume contract; also skip IDs already present in `failed.jsonl` unless `--retry-failed`).

- [ ] **Step 2: Pilot run — 10 tracks**

Run: `uv run python training/raveform_download.py --data-dir C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform --limit 10`

Then verify quality, not just existence:
- `librosa.load` the first 5 s of each downloaded mp3 (use the repo venv: `uv run python -c ...`) — nonzero samples, sane sample rate.
- Run ONE downloaded track through the sim end-to-end from the **main repo checkout** (it has the fast sim on master): `uv run python auto_pilot simulate file <data-dir>/audio/<id>.mp3 --report <scratch>/pilot_report.json` — must exit 0 or 1 (evaluation verdict), not crash. Note: this writes a decode-cache `.npy` beside the mp3 — expected, gitignored.
- Record pilot availability (n/10 succeeded) and mean seconds per track.

- [ ] **Step 3: Commit**

```bash
git add training/raveform_download.py
git commit -m "raveform: resumable yt-dlp audio downloader"
```

---

### Task 5: Launch the full download, detached

**Files:**
- Data only: `<data-dir>/download.log`, plus a copy of the downloader at `<data-dir>/raveform_download.py`

- [ ] **Step 1: Copy the script into the data dir**

Copy `training/raveform_download.py` and `manifest.csv`-reading dependencies (it should be a single self-contained file) to `<data-dir>/raveform_download.py`. Reason: the worktree is temporary; the detached process must not depend on it.

- [ ] **Step 2: Launch detached**

From PowerShell, launch the full run detached from this session, stdout+stderr to the log:

```powershell
Start-Process -WindowStyle Hidden -FilePath "python" `
  -ArgumentList "C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\raveform_download.py","--data-dir","C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform" `
  -RedirectStandardOutput "C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\download.log" `
  -RedirectStandardError "C:\Users\Julian\Projects\soundswitch-auto-pilot\training\data\raveform\download.err.log"
```

(Use the venv python if `librosa`-free; the downloader only needs stdlib + yt-dlp on PATH, so system `python` or the uv venv both work — state which you used.)

- [ ] **Step 3: Verify it is alive and progressing**

Wait ~2 minutes, then check `download.log` growth and `audio/` file count increasing beyond the pilot's 10. Confirm the process exists (`Get-Process python`). Estimate total ETA from the pilot's per-track time × remaining count and put it in your final report.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin raveform_data_pipeline
```

Do NOT open a PR — the coordinator reviews first.

---

## Final report (return this to the coordinator)

1. Annotation count, label vocabulary observed, per-label duration stats and transition counts (Task 3 output).
2. Pilot availability rate (n/10) and the sim-run verdict on the pilot track.
3. Full download: launch confirmation, current progress at report time, ETA, log locations, exact resume command.
4. yt-dlp/ffmpeg versions; any bot-check or throttling encountered (and that you did NOT work around it with cookies).
5. Anything that deviated from this plan and why.
