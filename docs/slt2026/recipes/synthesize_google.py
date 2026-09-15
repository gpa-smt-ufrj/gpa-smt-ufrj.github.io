"""
Synthesize 50 Portuguese sentences with Google Cloud TTS voices (pt-BR).

Voices (Neural2 — highest quality, Brazilian Portuguese):
  Male:   pt-BR-Neural2-B, pt-BR-Neural2-C
  Female: pt-BR-Neural2-A, pt-BR-Neural2-C (female variant)

Actually Google pt-BR Neural2 voices:
  pt-BR-Neural2-A  female
  pt-BR-Neural2-B  male
  pt-BR-Neural2-C  female
  pt-BR-Wavenet-E  male   (no Neural2-D male)

Using: Neural2-A (F), Neural2-C (F), Neural2-B (M), Wavenet-E (M)

Output: synth_wavs/google_{voice}/000..049.wav at 16kHz

Usage:
    python synthesize_google.py --key AIza...
"""
import argparse, base64, struct, time
import os
from pathlib import Path

import numpy as np
import requests

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SR      = 16000

VOICES = [
    # 5 female (Neural2 + Wavenet + Standard)
    ('google_neural2_a', 'pt-BR-Neural2-A', 'FEMALE'),
    ('google_neural2_c', 'pt-BR-Neural2-C', 'FEMALE'),
    ('google_wavenet_a', 'pt-BR-Wavenet-A', 'FEMALE'),
    ('google_wavenet_d', 'pt-BR-Wavenet-D', 'FEMALE'),
    ('google_standard_a','pt-BR-Standard-A','FEMALE'),
    # 5 male
    ('google_neural2_b', 'pt-BR-Neural2-B', 'MALE'),
    ('google_wavenet_b', 'pt-BR-Wavenet-B', 'MALE'),
    ('google_wavenet_c', 'pt-BR-Wavenet-C', 'MALE'),
    ('google_wavenet_e', 'pt-BR-Wavenet-E', 'MALE'),
    ('google_standard_b','pt-BR-Standard-B','MALE'),
]


def pcm_to_wav(pcm: bytes, sr: int = SR) -> bytes:
    wav  = b'RIFF'
    wav += struct.pack('<I', 36 + len(pcm))
    wav += b'WAVEfmt '
    wav += struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16)
    wav += b'data'
    wav += struct.pack('<I', len(pcm))
    wav += pcm
    return wav


def synthesize(text, voice_name, ssml_gender, api_key, retries=3):
    url = f'https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}'
    payload = {
        'input':       {'text': text},
        'voice':       {'languageCode': 'pt-BR', 'name': voice_name, 'ssmlGender': ssml_gender},
        'audioConfig': {'audioEncoding': 'LINEAR16', 'sampleRateHertz': SR},
    }
    for attempt in range(retries):
        r = requests.post(url, json=payload, timeout=60)
        if r.status_code == 200:
            audio_b64 = r.json()['audioContent']
            wav_bytes  = base64.b64decode(audio_b64)
            # Google returns full WAV; strip 44-byte header → raw PCM
            return wav_bytes[44:]
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
        else:
            print(f'    error {r.status_code}: {r.text[:200]}')
            time.sleep(2)
    raise RuntimeError('synthesis failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, help='Google Cloud API key')
    args = ap.parse_args()

    sentences = [s.strip() for s in open(os.environ.get('SYNTH_SENT_F', str(BASE / 'sentences_50.txt')), encoding='utf-8') if s.strip()]
    total = len(VOICES) * len(sentences)
    done  = 0

    for dir_name, voice_name, gender in VOICES:
        vdir = OUT_DIR / dir_name
        vdir.mkdir(parents=True, exist_ok=True)
        print(f'Voice: {voice_name} ({gender})')
        for i, sent in enumerate(sentences):
            out = vdir / f'{i:03d}.wav'
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                done += 1; continue
            print(f'  [{i+1:02d}/50] {sent[:60]} … ', end='', flush=True)
            pcm = synthesize(sent, voice_name, gender, args.key)
            out.write_bytes(pcm_to_wav(pcm))
            done += 1
            print(f'[{done}/{total}]')
            time.sleep(0.1)
        print()
    print('Done.')


if __name__ == '__main__':
    main()
