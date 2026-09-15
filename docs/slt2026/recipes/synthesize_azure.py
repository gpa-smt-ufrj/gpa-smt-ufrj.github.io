"""
Synthesize 50 Portuguese sentences with Azure Cognitive Services TTS (pt-BR).

Voices (Neural, Brazilian Portuguese):
  Male:   pt-BR-AntonioNeural, pt-BR-FabioNeural
  Female: pt-BR-FranciscaNeural, pt-BR-BrendaNeural

Output: synth_wavs/azure_{voice}/000..049.wav at 16kHz

Usage:
    python synthesize_azure.py --key <subscription_key> --region <region>
    e.g. --region eastus
"""
import argparse, struct, time
import os
from pathlib import Path

import requests

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SR      = 16000

VOICES = [
    # 5 male
    ('azure_antonio',  'pt-BR-AntonioNeural',  'Male'),
    ('azure_fabio',    'pt-BR-FabioNeural',    'Male'),
    ('azure_donato',   'pt-BR-DonatoNeural',   'Male'),
    ('azure_humberto', 'pt-BR-HumbertoNeural', 'Male'),
    ('azure_nicolau',  'pt-BR-NicolauNeural',  'Male'),
    # 5 female
    ('azure_francisca','pt-BR-FranciscaNeural','Female'),
    ('azure_brenda',   'pt-BR-BrendaNeural',   'Female'),
    ('azure_elza',     'pt-BR-ElzaNeural',     'Female'),
    ('azure_giovanna', 'pt-BR-GiovannaNeural', 'Female'),
    ('azure_leticia',  'pt-BR-LeticiaNeural',  'Female'),
]

SSML_TEMPLATE = """\
<speak version='1.0' xml:lang='pt-BR'>
  <voice name='{voice}'>{text}</voice>
</speak>"""


def pcm_to_wav(pcm: bytes, sr: int = SR) -> bytes:
    wav  = b'RIFF'
    wav += struct.pack('<I', 36 + len(pcm))
    wav += b'WAVEfmt '
    wav += struct.pack('<IHHIIHH', 16, 1, 1, sr, sr * 2, 2, 16)
    wav += b'data'
    wav += struct.pack('<I', len(pcm))
    wav += pcm
    return wav


def synthesize(text, voice_name, sub_key, region, retries=3):
    url = f'https://{region}.tts.speech.microsoft.com/cognitiveservices/v1'
    headers = {
        'Ocp-Apim-Subscription-Key': sub_key,
        'Content-Type': 'application/ssml+xml',
        'X-Microsoft-OutputFormat': 'riff-16khz-16bit-mono-pcm',
        'User-Agent': 'accent-research',
    }
    ssml = SSML_TEMPLATE.format(voice=voice_name, text=text)
    for attempt in range(retries):
        r = requests.post(url, headers=headers, data=ssml.encode('utf-8'), timeout=60)
        if r.status_code == 200:
            # Azure returns full WAV with 44-byte header; strip it
            return r.content[44:]
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
        else:
            print(f'    error {r.status_code}: {r.text[:200]}')
            time.sleep(2)
    raise RuntimeError('synthesis failed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--key',    required=True)
    ap.add_argument('--region', default='eastus')
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
            pcm = synthesize(sent, voice_name, args.key, args.region)
            out.write_bytes(pcm_to_wav(pcm))
            done += 1
            print(f'[{done}/{total}]')
            time.sleep(0.1)
        print()
    print('Done.')


if __name__ == '__main__':
    main()
