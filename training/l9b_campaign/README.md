# Nextgen retrain campaign -- PREP stage (decision #341)

CPU-only scripts. Outputs land in `<corpus>/models/l9b_campaign/` (CAMP);
the only corpus writes outside CAMP are `hand_label_admission.py`'s own
(beat grid, manifest row, clean row) and the armed `splits.json`. Nothing here
touches `segments.json`, checksums, `training/eval_*`, test-split membership,
or any benchmark baseline.

Stage order:

1. **`ng_prep.py`** -- enumerates the six new hand tracks (refuses on any
   surprise id), verifies full admission per id (hand.json / audio / beat grid /
   ok clean row), runs the documented admission path for anything missing, then
   writes `CAMP/extract_ids.txt` + `CAMP/audio_map.json` for
   `training.nn.ceiling.stream_extract` (phase-b tree; the extraction itself is
   the chain's GPU step, not this script's). `--post-extract` asserts all six
   F3 sidecars exist, load, and carry the causal F3/hop1 geometry.

2. **`ng_arm_splits.py`** -- ruling D1: pre-seeds all six into TRAIN via the
   frozen map (two would hash to test), then rebuilds through
   `training.nn.dataset.make_splits`. Dry-run by default; `--apply` backs up to
   `CAMP/splits.json.pre_nextgen.bak` first and asserts afterwards: six in
   train, val/test membership byte-unchanged, the 13 relabel ids still in
   train, exclusion lists unchanged. Any failure restores the backup, exit 1.

3. **`ng_build_overlay.py`** -- ruling D2: `CAMP/drop_demotion_overlay.json`,
   the published drop sections with `mean_db < -16.5` (-15.5 zero-false-
   demotion point + 1.0 dB margin), spans re-derived by in-order alignment to
   `segments.json` (count or >0.05 s duration drift is fatal), kept only if
   train-split, hand-free, outside test/val/eval. Also emits the report-only
   `CAMP/val_demotion_diagnostic.json` (never applied). Re-run post-arming
   with `--splits-file` if desired; published-id train membership does not
   change at arming.

4. **`ng_priors_refit.py`** -- per-arm priors on the label9 instrument.
   `--arm H` is the plain train-split fit (proven byte-identical to running
   label9's `priors.py` directly: `--verify-anchor`, artifacts under
   `CAMP/anchor/`). `--arm HD` rewrites the overlay's drop sections to
   breakdown before fitting and prints the floors/transition delta vs H;
   refuses an overlay id outside the split or a span matching no published
   drop section. Re-run both arms post-arming.

`ng_common.py` holds the shared paths, the chartered six ids, and the D2
thresholds.
