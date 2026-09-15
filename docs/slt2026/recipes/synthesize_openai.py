"""
Synthesize 50 Portuguese sentences with OpenAI TTS voices.

Voices (all support multilingual/Portuguese):
  Male:   onyx, echo, fable
  Female: nova, shimmer, alloy
  → picking top 2M + 2F for consistency with experiment design:
    onyx (male), echo (male), nova (female), shimmer (female)

Model: tts-1-hd (highest quality)
Output: synth_wavs/openai_{voice}/000..049.wav  at 16kHz

Usage:
    python synthesize_openai.py --key sk-...
"""
import argparse, struct, time
import os
from pathlib import Path

import numpy as np
import requests

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SR      = 16000
MODEL   = 'tts-1-hd'

VOICES = [
    # 5 male
    ('openai_onyx',    'onyx',    'male'),
    ('openai_echo',    'echo',    'male'),
    ('openai_fable',   'fable',   'male'),
    ('openai_ash',     'ash',     'male'),
    # 5 female (openai has 4M + 5F = 9 total, no 5th male voice)
    ('openai_nova',    'nova',    'female'),
    ('openai_shimmer', 'shimmer', 'female'),
    ('openai_alloy',   'alloy',   'female'),
    ('openai_coral',   'coral',   'female'),
    ('openai_sage',    'sage',    'female'),
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


def synthesize(text, voice_name, api_key, retries=3):
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    payload = {'model': MODEL, 'voice': voice_name, 'input': text,
               'response_format': 'pcm'}   # raw 24kHz PCM
    for attempt in range(retries):
        r = requests.post('https://api.openai.com/v1/audio/speech',
                          headers=headers, json=payload, timeout=60)
        if r.status_code == 200:
            # OpenAI PCM is 24kHz; downsample to 16kHz
            import numpy as np, librosa
            audio = np.frombuffer(r.content, dtype=np.int16).astype(np.float32) / 32768.0
            audio = librosa.resample(audio, orig_sr=24000, target_sr=SR)
            return (audio * 32768).astype(np.int16).tobytes()
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
        else:
            print(f'    error {r.status_code}: {r.text[:200]}')
            time.sleep(2)
    raise RuntimeError('synthesis failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key', required=True, help='OpenAI API key')
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
            try:
                pcm = synthesize(sent, voice_name, args.key)
            except Exception as e:
                print(f'SKIP (failed): {e}')
                continue
            out.write_bytes(pcm_to_wav(pcm))
            done += 1
            print(f'[{done}/{total}]')
            time.sleep(0.2)
        print()
    print('Done.')


if __name__ == '__main__':
    main()
