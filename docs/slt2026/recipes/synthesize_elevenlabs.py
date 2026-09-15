"""
Synthesize 50 Portuguese sentences with 4 ElevenLabs voices (2F + 2M).
Saves 200 WAV files at 16kHz to synth_wavs/{voice_name}/{000..049}.wav.

Voices chosen by popularity (top premade):
  Female: Sarah (EXAVITQu4vr4xnSDxMaL), Laura (FGY2WhTYpPnrIDTdsKH5)
  Male:   Brian (nPczCjzI2devNBz1zQrb), George (JBFqnCBsd6RMkjVDRZzb)
"""
import time
import struct
import os
from pathlib import Path

import numpy as np
import requests

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
APIKEY  = os.environ.get("ELEVENLABS_API_KEY", "")
MODEL   = "eleven_multilingual_v2"
OUT_DIR = Path(os.environ.get("SYNTH_OUT_DIR", str(BASE / "synth_wavs")))
SR      = 16000

VOICES = [
    # Top 5 male Brazilian Portuguese voices by usage
    ("elevenlabs_Matheus",     "36rVQA1AOIPwpA3Hg1tC", "male"),    # 711k
    ("elevenlabs_Lax",         "tS45q0QcrDHqHoaWdCDR", "male"),    # 241k
    ("elevenlabs_Will",        "CstacWqMhJQlnfLPxRG4", "male"),    # 177k
    ("elevenlabs_Adriano",     "hwnuNyWkl9DjdTFykrN6", "male"),    # 119k
    ("elevenlabs_VictorPower", "YNOujSUmHtgN6anjqXPf", "male"),    # 114k
    # Top 5 female Brazilian Portuguese voices by usage
    ("elevenlabs_Keren",       "33B4UnXyTNbgLmdEDh5P", "female"),  # 116k
    ("elevenlabs_YasminAlves", "lWq4KDY8znfkV0DrK8Vb", "female"),  # 107k
    ("elevenlabs_Carla",       "oJebhZNaPllxk6W0LSBA", "female"),  # 82k
    ("elevenlabs_Roberta",     "RGymW84CSmfVugnA5tvA", "female"),  # 64k
    ("elevenlabs_Scheila",     "cyD08lEy76q03ER1jZ7y", "female"),  # 62k
]

HEADERS = {
    "xi-api-key":   APIKEY,
    "Content-Type": "application/json",
}

VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}


def pcm_to_wav_bytes(pcm: bytes, sr: int = SR) -> bytes:
    n_samples = len(pcm) // 2
    wav  = b"RIFF"
    wav += struct.pack("<I", 36 + len(pcm))
    wav += b"WAVEfmt "
    wav += struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    wav += b"data"
    wav += struct.pack("<I", len(pcm))
    wav += pcm
    return wav


def synthesize(text: str, voice_id: str, retries: int = 3) -> bytes:
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
           f"?output_format=pcm_16000")
    payload = {"text": text, "model_id": MODEL, "voice_settings": VOICE_SETTINGS}
    for attempt in range(retries):
        r = requests.post(url, headers=HEADERS, json=payload, timeout=60)
        if r.status_code == 200:
            return r.content
        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    rate-limited, waiting {wait}s …")
            time.sleep(wait)
        else:
            print(f"    error {r.status_code}: {r.text[:200]}")
            time.sleep(2)
    raise RuntimeError(f"Failed after {retries} attempts")


def main():
    sentences = [s.strip() for s in
                 open(os.environ.get("SYNTH_SENT_F", str(BASE / "sentences_50.txt")), encoding="utf-8")
                 if s.strip()]
    print(f"Sentences: {len(sentences)}")
    print(f"Voices:    {len(VOICES)}")
    print(f"Total WAVs: {len(sentences) * len(VOICES)}\n")

    total = len(voices := VOICES) * len(sentences)
    done  = 0

    for vname, vid, vgender in voices:
        vdir = OUT_DIR / vname
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"Voice: {vname} ({vgender})")

        for i, sent in enumerate(sentences):
            out = vdir / f"{i:03d}.wav"
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                print(f"  [{i+1:02d}/50] already exists, skipping")
                done += 1
                continue

            print(f"  [{i+1:02d}/50] {sent[:60]}", end=" … ", flush=True)
            try:
                pcm = synthesize(sent, vid)
            except Exception as e:
                print(f"SKIP (failed): {e}")
                continue
            wav = pcm_to_wav_bytes(pcm)
            out.write_bytes(wav)
            done += 1
            print(f"{len(pcm)//2/SR:.1f}s  [{done}/{total}]")
            time.sleep(0.3)   # gentle pacing

        print()

    print("Done.")


if __name__ == "__main__":
    main()
