# L9C campaign drivers (#346)

The overnight retrain on the owner's completed slow-climb label session.
Artifacts land in `<corpus>/models/l9c_campaign/` (CAMP); the only corpus
writes are inside CAMP. Nothing here touches `segments.json`, checksums,
`training/eval_*`, splits.json, val/test membership, annotations, or any
benchmark baseline (arm N-IW trains via a campaign-local `splits_niw.json`).

Two disclosed differences from the l9b drivers these were cut from:

1. **The instrument imports from the MAIN repo**, not the label9 worktree —
   the #345 `buildup_drop_bonus` knob lives in `training/nn/decoder.py`.
   Parity is gated by anchors, never assumed: every verdict run must
   reproduce the l9 campaign's banked row AND both l9b ng_H rows to all
   digits before any new row is read.
2. **The #344 transition instrument**: every verdict row also reports
   buildup-preceded-drop (decoded class of the bar before each matched drop
   entry, ±2.0 s) and the buildup entry-lag distribution (median/p90 +
   never-entered). Reported beside the class scores, not gated.

Drivers, in stage order:

- `ng_priors_refit.py` — arm N priors from the merged annotation view;
  `--verify-anchor` byte-compares the label9 and main trees' CLI fits.
- `ng_sweep_driver.py` — the per-arm two-seed sweep, grid extended with the
  pre-registered transition axis (`buildup_drop_bonus` 0.0/0.4/0.7/1.1/1.6);
  selection rule unchanged from l9b. `--smoke` proves the machinery on l9
  posteriors.
- `ng_decoded_verdict.py` — the board: l9 + l9b ng_H anchors + the l9c arms,
  each under its own registered decode instrument, with the #344 columns.
- `ng_probe_rig.py` — shadow-corpus probe sims (mirrors the SHIPPED
  models/l9b generation), the #344 read lines, fit-diagnostic tracks
  (inyathi/yai/skylark) beside the two owner probes.
- `ng_arc_read.py` — the #348 trade-off table: arcs entered/held and the
  strict drop clause, recomputed from the banked probe report JSONs. Versioned
  here because the ops copy produced a false finding — it asked which block
  STARTED in the labelled window, so a chain already holding DROP scored as
  having lost a drop it lit completely. It now reports SECONDS in force over
  the labelled span (a boolean alone is wrong in both directions), and keeps
  the superseded reading under `first_entry_*` so banked digits stay
  comparable. `tests/test_arc_read.py` pins both.
- `ng_build_mask_overlay.py` / `ng_mask_proof.py` — the N-MK conditional:
  the l9b mask re-mined minus hand-labeled tracks, proven zero-loss before
  any train.
- `ng_state.py`, `run_train.ps1` — the campaign state machine (heartbeats +
  nonpaged-pool gate) and the BelowNormal train launcher (ops copies run
  from CAMP).

The night's verdict, board, probe reads and the owner's morning report are
in CAMP (`L9C_RESULTS.md`, `NG_DECODED.md`, `probe_reads/`); the
pre-registrations and executed verdicts are #346 addenda in the local
decisions log.
