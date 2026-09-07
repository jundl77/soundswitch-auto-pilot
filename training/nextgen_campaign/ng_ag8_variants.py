"""ARM H-AG8 (#343 rung 5): build the pitch-shift variants of the six hand tracks.

Decodes each base corpus mp3 with the extractor's own ffmpeg arithmetic
(44.1 kHz mono f32), pitch-shifts it per semitone step with librosa
(STFT phase-vocoder -- TIME-PRESERVING ONLY, no tempo/stretch, so every label
instant copies verbatim; sample-count equality is asserted, never assumed),
and writes float32 WAVs under CAMP/aug_audio/.  Emits extract_ids_ag8.txt +
audio_map_ag8.json in the exact format stream_extract's --ids-file and
--audio-map consume.  Campaign-local only: corpus audio/ is never touched.
Resumable: an existing WAV of the right sample count is kept.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys

import numpy as np

from ng_common import CAMP, CORPUS, NEW_HAND_IDS

try:
    _k32 = ctypes.windll.kernel32
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.SetPriorityClass(_k32.GetCurrentProcess(), 0x00004000)
except Exception:  # noqa: BLE001
    pass

SAMPLE_RATE = 44100
SHIFTS = (("m3", -3), ("m2", -2), ("m1", -1),
          ("p1", 1), ("p2", 2), ("p3", 3), ("p4", 4))
AUG_DIR = CAMP / "aug_audio"
FEAT_ROOT = CAMP / "features_ag8_root" / "MERT-v1-330M_L6-22_F3_hop1"


def variant_id(base: str, suffix: str) -> str:
    return f"{base}__ps{suffix}"


def decode(path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
         "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def main() -> int:
    import librosa
    import soundfile as sf

    AUG_DIR.mkdir(parents=True, exist_ok=True)
    audio_map: dict = {}
    to_extract: list = []
    total_sec = 0.0
    for base in NEW_HAND_IDS:
        mp3 = CORPUS / "audio" / f"{base}.mp3"
        if not mp3.exists():
            raise SystemExit(f"missing base audio {mp3}")
        y = None
        for suffix, steps in SHIFTS:
            vid = variant_id(base, suffix)
            wav = AUG_DIR / f"{vid}.wav"
            if wav.exists():
                info = sf.info(str(wav))
                if y is None:
                    y = decode(mp3)
                if info.frames == len(y) and info.samplerate == SAMPLE_RATE:
                    print(f"{vid}: kept existing WAV ({info.frames} samples)")
                else:
                    raise SystemExit(f"{wav} exists with wrong geometry "
                                     f"({info.frames} vs {len(y)}) -- refusing")
            else:
                if y is None:
                    y = decode(mp3)
                shifted = librosa.effects.pitch_shift(
                    y, sr=SAMPLE_RATE, n_steps=steps)
                if len(shifted) != len(y):
                    raise SystemExit(f"{vid}: pitch_shift moved the sample "
                                     f"count {len(y)} -> {len(shifted)}")
                sf.write(str(wav), shifted.astype(np.float32), SAMPLE_RATE,
                         subtype="FLOAT")
                peak = float(np.abs(shifted).max())
                print(f"{vid}: {len(shifted)} samples "
                      f"({len(shifted) / SAMPLE_RATE:.1f} s), peak {peak:.3f}",
                      flush=True)
            total_sec += len(y) / SAMPLE_RATE
            audio_map[vid] = str(wav.resolve())
            if not (FEAT_ROOT / f"{vid}.npz").exists():
                to_extract.append(vid)

    lines = ["# ARM H-AG8 (#343): pitch-shift variants missing an F3 sidecar",
             "# consumed by training.nn.ceiling.stream_extract --ids-file"]
    (CAMP / "extract_ids_ag8.txt").write_text(
        "\n".join(lines + to_extract) + "\n", encoding="utf-8")
    (CAMP / "audio_map_ag8.json").write_text(
        json.dumps(audio_map, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(audio_map)} variants, {total_sec:.0f} s of audio; "
          f"{len(to_extract)} still need F3 sidecars")
    print(f"wrote {CAMP / 'extract_ids_ag8.txt'}")
    print(f"wrote {CAMP / 'audio_map_ag8.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
