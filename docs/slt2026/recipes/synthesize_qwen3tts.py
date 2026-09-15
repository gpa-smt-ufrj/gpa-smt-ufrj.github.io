"""
Synthesize the 50 pt-BR sentences using Qwen3-TTS.
Model: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice

Two preset voices (one male, one female).

Output:
  synth_wavs/qwen3_ryan/000..049.wav    (male, preset 'Ryan')
  synth_wavs/qwen3_bella/000..049.wav   (female, preset 'Bella')

Usage:
    python synthesize_qwen3tts.py
    # optional: --voices ryan bella  (to pick specific presets)
"""
import argparse, wave
import os
from pathlib import Path

import numpy as np
import torch
import soundfile as sf

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SENT_F  = Path(os.environ.get('SYNTH_SENT_F', str(BASE / 'sentences_50.txt')))
SR_OUT  = 16000

VOICES = [
    ('qwen3_aiden',    'aiden',    'male'),
    ('qwen3_dylan',    'dylan',    'male'),
    ('qwen3_eric',     'eric',     'male'),
    ('qwen3_ono_anna', 'ono_anna', 'female'),
    ('qwen3_ryan',     'ryan',     'male'),
    ('qwen3_serena',   'serena',   'female'),
    ('qwen3_sohee',    'sohee',    'female'),
    ('qwen3_uncle_fu', 'uncle_fu', 'male'),
    ('qwen3_vivian',   'vivian',   'female'),
]


def save_wav(audio: np.ndarray, path: Path, sr: int = SR_OUT):
    if audio.dtype != np.int16:
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def resample_if_needed(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR_OUT:
        return audio
    import torchaudio
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, src_sr, SR_OUT)
    return t.squeeze(0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voices', nargs='+', default=None,
                    help='Which preset voices to use (default: ryan bella)')
    args = ap.parse_args()

    sentences = [s.strip() for s in SENT_F.read_text('utf-8').splitlines() if s.strip()]
    assert len(sentences) >= 1

    selected = VOICES
    if args.voices:
        names_lower = {v.lower() for v in args.voices}
        selected = [(d, v, g) for d, v, g in VOICES if v.lower() in names_lower]

    print('Loading Qwen3-TTS-12Hz-1.7B-CustomVoice...')
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        device_map='cuda:0' if torch.cuda.is_available() else 'cpu',
        dtype=torch.bfloat16,
    )
    print('Model loaded.')

    for voice_dir, speaker, gender in selected:
        vdir = OUT_DIR / voice_dir
        vdir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== {voice_dir} (speaker={speaker}, {gender}) ===')

        for i, sent in enumerate(sentences):
            out = vdir / f'{i:03d}.wav'
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                print(f'  [{i+1:02d}/50] skip (exists)')
                continue

            print(f'  [{i+1:02d}/50] {sent[:60]}…', end=' ', flush=True)
            try:
                wavs, sr = model.generate_custom_voice(
                    text=sent,
                    language='Portuguese',
                    speaker=speaker,
                    instruct='',
                )
                audio = np.array(wavs[0])
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)
                audio = resample_if_needed(audio.astype(np.float32), sr)
                save_wav(audio, out)
                print('ok')
            except Exception as e:
                print(f'ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
