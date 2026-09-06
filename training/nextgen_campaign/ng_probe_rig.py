"""The probe-simulation rig for the nextgen campaign (#341).

Runs a named student chain over the probe tracks through the FULL fast
simulation.  The sim resolves its model artifacts through
training/corpus_root.py, so an arm chain is a SHADOW corpus root under
CAMP/shadow_<chain>/ (models/l9/ mirroring the real layout, with the arm's
student export, priors and decoder config swapped in) selected via
$RAVEFORM_DATA_DIR.

The decoder config is the one artifact the chain does NOT read off the corpus
root: lib.engine.section_decoder loads nn.decoder.SHIPPING_DECODER_CONFIG,
which lives beside nn/decoder.py itself.  The rig therefore ships a shadow
``nn`` package (byte-copies of the repo's decoder.py/priors.py plus the
chain's decoder_config.json) and pre-imports it in the launcher before
lib.engine.section_decoder can insert the repo's training dir -- an already-
imported ``nn`` wins over any later sys.path insert.  The anchor subcommand
proves the whole mechanism on a committed, warm (pure-CPU) eval-set track:
direct run vs shadow run must be byte-identical.

Timeline reads are a fit/expressiveness diagnostic: both probe tracks are
train-split hand labels, so nothing here is a held-out score.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAIN = Path(r"C:\Users\Julian\Projects\soundswitch-auto-pilot")
DATA = MAIN / "training" / "data" / "raveform"
CAMP = DATA / "models" / "nextgen_campaign"
L9 = DATA / "models" / "l9"
MODEL_VERSION = "l9_w128_s1234"
PHASE_B = Path(r"C:\Users\Julian\Projects\soundswitch-phase-b-worktree")
CEILING_PY = Path(r"C:\Users\Julian\Projects\soundswitch-exp-ceiling-worktree"
                  r"\.venv\Scripts\python.exe")
ANCHOR_TRACK = MAIN / "training" / "eval_audio" / "0yMOLeJRKr.mp3"

TRACKS = {
    "opus": DATA / "audio" / "hand-65cb8c94812d.mp3",
    "dwmu": DATA / "audio" / "hand-8339586c555a.mp3",
}
HAND_IDS = {"opus": "hand-65cb8c94812d", "dwmu": "hand-8339586c555a"}

INTENT_FOR_CLASS = {
    "intro": "atmospheric", "altintro": "atmospheric", "buildup": "buildup",
    "breakdown": "breakdown", "bridge": "breakdown", "drop": "drop",
    "cooldown": "breakdown", "outro": "atmospheric", "altoutro": "atmospheric",
}
SEARCH_LEAD_SEC = 8.0

RECORD_GEOMETRY_FIELDS = ("window_cells", "input_dim", "rnn_hidden",
                          "future_cells", "future_sec", "label_frame_sec",
                          "sha256")

# Which arm's priors/decoder-config a chain decodes under.  HOS (#342) trains
# on arm H's labels with six tracks oversampled, so its labels -- and
# therefore its priors and first-read config -- are arm H's own files; only
# the student export differs.
CHAIN_ARTIFACTS = {"H": "H", "HD": "HD", "HOS": "H"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shadow_root(chain: str) -> Path:
    return CAMP / f"shadow_{chain}"


def _junction_or_copy(link: Path, target: Path) -> str:
    if link.exists():
        return "existed"
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          capture_output=True, text=True)
    if proc.returncode == 0:
        return "junction"
    shutil.copytree(target, link)
    return "copied"


def build_shadow(chain: str) -> Path:
    root = shadow_root(chain)
    generation = root / "models" / "l9"
    generation.mkdir(parents=True, exist_ok=True)

    for source in sorted(L9.iterdir()):
        if not source.is_file():
            continue
        target = generation / source.name
        label = CHAIN_ARTIFACTS.get(chain)
        if label:
            if source.name == "priors.json":
                shutil.copy2(CAMP / f"priors_{label}.json", target)
                continue
            if source.name == "decoder_config.json":
                shutil.copy2(CAMP / f"decoder_config_{label}.json", target)
                continue
        shutil.copy2(source, target)

    how = _junction_or_copy(generation / "bar_tracker", L9 / "bar_tracker")

    run_dir = generation / MODEL_VERSION
    run_dir.mkdir(exist_ok=True)
    if chain == "anchor":
        for source in sorted((L9 / MODEL_VERSION).iterdir()):
            shutil.copy2(source, run_dir / source.name)

    nn_dir = root / "nn_shadow" / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "decoder.py", "priors.py"):
        shutil.copy2(MAIN / "training" / "nn" / name, nn_dir / name)
    config_source = (MAIN / "training" / "nn" / "decoder_config.json"
                     if chain == "anchor"
                     else CAMP / f"decoder_config_{CHAIN_ARTIFACTS[chain]}.json")
    shutil.copy2(config_source, nn_dir / "decoder_config.json")

    print(f"shadow {root} built (bar_tracker: {how}; decoder config: "
          f"{config_source})")
    if chain in CHAIN_ARTIFACTS and not (run_dir / "online_step.onnx").exists():
        print(f"NOTE: {run_dir / 'online_step.onnx'} absent -- run export-arm "
              f"before simulating this chain")
    return root


def export_arm(chain: str, run: str) -> None:
    root = shadow_root(chain)
    out = root / "models" / "l9" / MODEL_VERSION / "online_step.onnx"
    checkpoint = CAMP / run / "best.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"no checkpoint at {checkpoint}")
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [str(CEILING_PY), "-m", "training.nn.ceiling.online_export", "export",
         "--checkpoint", str(checkpoint), "--out", str(out)],
        cwd=str(PHASE_B), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"online_export failed with rc {proc.returncode}")

    record = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    missing = [f for f in RECORD_GEOMETRY_FIELDS if f not in record]
    if missing:
        raise RuntimeError(f"exported record lacks {missing}")
    if record["sha256"] != sha256_file(out):
        raise RuntimeError("exported record sha256 does not match the graph")
    if not record.get("verification", {}).get("agrees"):
        raise RuntimeError("export verification does not agree")
    shipped = json.loads((L9 / MODEL_VERSION / "online_step.onnx.json")
                         .read_text(encoding="utf-8"))
    for field in ("window_cells", "input_dim", "rnn_hidden", "future_cells",
                  "future_sec", "label_frame_sec"):
        if record[field] != shipped[field]:
            raise RuntimeError(
                f"arm geometry {field}={record[field]} != shipped "
                f"{shipped[field]} -- the shadow chain would run a different "
                f"stream geometry than the one the affine was fitted on")
    print(f"exported {out} (sha {record['sha256'][:12]}), record format and "
          f"geometry match the shipped {MODEL_VERSION} record")


def _warm(track: Path) -> bool:
    return ((track.parent / f"{track.name}.librosa.mertcells.npz").exists()
            and (track.parent / f"{track.name}.librosa.bartracker.npz").exists())


def run_sim(chain: str, track: Path, report: Path, *,
            allow_cold: bool = False, hide_cuda: bool = False) -> int:
    if not _warm(track) and not allow_cold:
        raise RuntimeError(
            f"{track.name} has no warm cell+tracker sidecars -- this run "
            f"would need the GPU; pass --allow-cold when the card is free")
    report.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.pop("RAVEFORM_DATA_DIR", None)
    if hide_cuda:
        env["CUDA_VISIBLE_DEVICES"] = ""
    if chain == "shipped":
        command = [sys.executable, str(MAIN / "auto_pilot"), "simulate",
                   "file", str(track), "--report", str(report)]
    else:
        root = shadow_root(chain)
        if not (root / "models" / "l9" / MODEL_VERSION
                / "online_step.onnx").exists():
            raise RuntimeError(f"shadow {root} has no student export")
        env["RAVEFORM_DATA_DIR"] = str(root)
        command = [sys.executable, str(Path(__file__).resolve()), "_run-sim",
                   "--shadow", str(root), str(track), str(report)]
    proc = subprocess.run(command, cwd=str(MAIN), env=env)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"simulation died with rc {proc.returncode}")
    if not report.exists():
        raise RuntimeError(f"simulation wrote no report at {report}")
    print(f"[{chain}] report {report} (sim rc {proc.returncode}, "
          f"sha256 {sha256_file(report)[:16]})")
    return proc.returncode


def run_sim_inner(shadow: str, track: str, report: str) -> None:
    root = Path(shadow)
    sys.path[:0] = [str(root / "nn_shadow"), str(MAIN)]
    import nn.decoder  # noqa: PLC0415

    wanted = (root / "nn_shadow" / "nn" / "decoder_config.json").resolve()
    loaded = Path(nn.decoder.SHIPPING_DECODER_CONFIG).resolve()
    if loaded != wanted:
        raise RuntimeError(f"shadow nn package did not win: decoder config "
                           f"resolves to {loaded}, wanted {wanted}")
    os.environ["RAVEFORM_DATA_DIR"] = str(root)
    sys.argv = ["auto_pilot", "simulate", "file", track, "--report", report]
    from lib.main import main  # noqa: PLC0415

    asyncio.new_event_loop().run_until_complete(main())


def _classifier_blocks(report: dict) -> list:
    blocks = [b for b in report["intents"] if "song_t" in b]
    blocks.sort(key=lambda b: b["song_t"])
    return blocks


def _first_entry(blocks, intent, lo, hi):
    for block in blocks:
        if block["trigger"] == "silence":
            continue
        if lo <= block["song_t"] < hi and block["intent"] == intent:
            return block
    return None


def _intent_at(blocks, t):
    current = None
    for block in blocks:
        if block["song_t"] <= t:
            current = block
        else:
            break
    return current


def load_hand_sections(track_key: str) -> list:
    record = json.loads((DATA / "annotations" / f"{HAND_IDS[track_key]}.hand.json")
                        .read_text(encoding="utf-8"))
    return [(s["name"], float(s["start"]), float(s["end"]))
            for s in record["sections"]]


def timeline_read(chain: str, track_key: str, report_path: Path,
                  sections) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    blocks = _classifier_blocks(report)

    lines = [f"# probe read: chain={chain} track={track_key}",
             "",
             f"report: {report_path}  (sha256 {sha256_file(report_path)[:16]})",
             "fit/expressiveness diagnostic -- train-split hand labels, not a "
             "held-out score", ""]

    lines += ["## committed intent timeline (song_t, intent, trigger)", ""]
    for block in blocks:
        lines.append(f"- {block['song_t']:8.2f}s  {block['intent']:<12} "
                     f"[{block['trigger']}]")
    lines.append("")

    if sections is None:
        lines += ["(no hand labels for this track -- parse mechanics only)", ""]
        return "\n".join(lines)

    lines += ["## label vs show", "",
              "| label section | start | end | intent wanted | entered at "
              "(delta) | held to section end |",
              "|---|---|---|---|---|---|"]
    for name, start, end in sections:
        wanted = INTENT_FOR_CLASS[name]
        entry = _first_entry(blocks, wanted, start - SEARCH_LEAD_SEC, end)
        if entry is None:
            entered = "never"
        else:
            entered = f"{entry['song_t']:.2f}s ({entry['song_t'] - start:+.2f}s)"
        at_end = _intent_at(blocks, max(start, end - 0.5))
        held = "-" if entry is None else (
            "yes" if at_end is not None and at_end["intent"] == wanted else
            f"no ({at_end['intent'] if at_end else 'none'} at end)")
        lines.append(f"| {name} | {start:.2f} | {end:.2f} | {wanted} | "
                     f"{entered} | {held} |")
    lines.append("")

    drops = [(s, e) for name, s, e in sections if name == "drop"]
    if drops:
        lines += ["## drop landings", ""]
        for n, (start, end) in enumerate(drops, 1):
            entry = _first_entry(blocks, "drop", start - SEARCH_LEAD_SEC, end)
            if entry is None:
                lines.append(f"- drop {n} at {start:.2f}s: DROP never landed "
                             f"inside the section")
            else:
                lines.append(f"- drop {n} at {start:.2f}s: DROP landed at "
                             f"{entry['song_t']:.2f}s "
                             f"({entry['song_t'] - start:+.2f}s)")
        lines.append("")
    return "\n".join(lines)


def do_read(chain: str, track_key: str, report_path: Path,
            with_labels: bool) -> Path:
    sections = load_hand_sections(track_key) if with_labels else None
    text = timeline_read(chain, track_key, report_path, sections)
    out_dir = CAMP / "probe_reads"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{chain}_{track_key}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out}")
    return out


def combine() -> None:
    out_dir = CAMP / "probe_reads"
    parts = [path.read_text(encoding="utf-8")
             for path in sorted(out_dir.glob("*_*.md"))
             if path.name != "PROBES.md"]
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out_dir / "PROBES.md").write_text(
        f"# nextgen campaign probe reads ({stamp})\n\n" + "\n---\n\n".join(parts),
        encoding="utf-8")
    print(f"wrote {out_dir / 'PROBES.md'}")


def anchor() -> int:
    build_shadow("anchor")
    out_dir = CAMP / "probe_anchor"
    out_dir.mkdir(parents=True, exist_ok=True)
    direct = out_dir / "direct.json"
    shadow = out_dir / "shadow.json"
    run_sim("shipped", ANCHOR_TRACK, direct, hide_cuda=True)
    run_sim("anchor", ANCHOR_TRACK, shadow, hide_cuda=True)
    sha_direct, sha_shadow = sha256_file(direct), sha256_file(shadow)
    identical = sha_direct == sha_shadow
    verdict = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "track": str(ANCHOR_TRACK),
        "direct": {"report": str(direct), "sha256": sha_direct},
        "shadow": {"report": str(shadow), "sha256": sha_shadow,
                   "root": str(shadow_root("anchor"))},
        "byte_identical": identical,
        "mechanism": "RAVEFORM_DATA_DIR shadow corpus root + pre-imported "
                     "shadow nn package (the decoder-config swap seam), "
                     "nothing swapped -- the rig's identity anchor",
    }
    (out_dir / "ANCHOR.json").write_text(json.dumps(verdict, indent=2) + "\n",
                                         encoding="utf-8")
    print(f"direct  {sha_direct}")
    print(f"shadow  {sha_shadow}")
    print("ANCHOR BYTE-IDENTICAL" if identical else "ANCHOR MISMATCH")
    return 0 if identical else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-shadow")
    p.add_argument("--chain", choices=("anchor", "H", "HD", "HOS"),
                   required=True)

    p = sub.add_parser("export-arm")
    p.add_argument("--arm", choices=("H", "HD", "HOS"), required=True)
    p.add_argument("--run", default=None)

    p = sub.add_parser("sim")
    p.add_argument("--chain", choices=("shipped", "anchor", "H", "HD", "HOS"),
                   required=True)
    p.add_argument("--track", required=True, help="opus | dwmu | path")
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--allow-cold", action="store_true")

    p = sub.add_parser("read")
    p.add_argument("--chain", required=True)
    p.add_argument("--track", choices=tuple(TRACKS), required=True)
    p.add_argument("--report", type=Path, default=None)

    p = sub.add_parser("parse-check")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--dry-labels", choices=tuple(TRACKS), default=None)

    sub.add_parser("combine")
    sub.add_parser("anchor")

    p = sub.add_parser("_run-sim")
    p.add_argument("--shadow", required=True)
    p.add_argument("track")
    p.add_argument("report")

    args = parser.parse_args()
    if args.command == "_run-sim":
        run_sim_inner(args.shadow, args.track, args.report)
        return 0
    if args.command == "build-shadow":
        build_shadow(args.chain)
        return 0
    if args.command == "export-arm":
        export_arm(args.arm, args.run or f"ng_{args.arm}_w128_s1234")
        return 0
    if args.command == "anchor":
        return anchor()
    if args.command == "combine":
        combine()
        return 0
    if args.command == "sim":
        track_key = args.track if args.track in TRACKS else None
        track = TRACKS.get(args.track, Path(args.track))
        report = args.report or (CAMP / "probe_reports"
                                 / f"{args.chain}_{track_key or track.stem}.json")
        run_sim(args.chain, track, report, allow_cold=args.allow_cold)
        if track_key is not None:
            do_read(args.chain, track_key, report, with_labels=True)
        return 0
    if args.command == "read":
        report = args.report or (CAMP / "probe_reports"
                                 / f"{args.chain}_{args.track}.json")
        do_read(args.chain, args.track, report, with_labels=True)
        return 0
    if args.command == "parse-check":
        report = json.loads(args.report.read_text(encoding="utf-8"))
        blocks = _classifier_blocks(report)
        print(f"{args.report}: {len(blocks)} intent blocks with song_t; "
              f"first {blocks[0] if blocks else None}")
        text = timeline_read("parse-check", "opus", args.report, None)
        print(text)
        if args.dry_labels:
            sections = load_hand_sections(args.dry_labels)
            print(f"hand labels {args.dry_labels}: {len(sections)} sections")
            for name, start, end in sections:
                print(f"  {name:<10} {start:8.2f} -> {end:8.2f} "
                      f"(intent {INTENT_FOR_CLASS[name]})")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
