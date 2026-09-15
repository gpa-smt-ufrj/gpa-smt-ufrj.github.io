"""
Synthesize the 50 pt-BR sentences using Piper TTS Brazilian Portuguese voices.

Available pt-BR voices:
  pt_BR-cadu-medium    (male)
  pt_BR-edresson-low   (male)
  pt_BR-faber-medium   (male)
  pt_BR-jeff-medium    (male)

Model files are auto-downloaded to  ~/.local/share/piper-tts/

Output:  synth_wavs/piper_{name}/000..049.wav

Usage:
    python synthesize_piper_ptbr.py
"""
import io, struct, wave
import os
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SENT_F  = Path(os.environ.get('SYNTH_SENT_F', str(BASE / 'sentences_50.txt')))
SR_OUT  = 16000

MODEL_DIR = Path.home() / '.local' / 'share' / 'piper-tts'

VOICES = [
    ('piper_cadu',     'pt_BR-cadu-medium',    'male'),
    ('piper_edresson', 'pt_BR-edresson-low',   'male'),
    ('piper_faber',    'pt_BR-faber-medium',   'male'),
    ('piper_jeff',     'pt_BR-jeff-medium',    'male'),
]


def download_voice(voice_id: str) -> tuple[Path, Path]:
    """Download piper onnx + config if missing. Returns (onnx_path, config_path)."""
    from piper.download_voices import download_voice as piper_dl
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    onnx = MODEL_DIR / f'{voice_id}.onnx'
    cfg  = MODEL_DIR / f'{voice_id}.onnx.json'
    if not onnx.exists() or not cfg.exists():
        print(f'  Downloading {voice_id}…', end=' ', flush=True)
        piper_dl(voice_id, MODEL_DIR)
        print('done')
    return onnx, cfg


def save_wav_pcm(pcm_bytes: bytes, path: Path, src_sr: int):
    """Save raw PCM (int16 mono) to a 16-kHz WAV, resampling if needed."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    if src_sr != SR_OUT:
        import torchaudio, torch
        t = torch.from_numpy(arr).unsqueeze(0)
        arr = torchaudio.functional.resample(t, src_sr, SR_OUT).squeeze(0).numpy()
    pcm16 = np.clip(arr, -1, 1)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR_OUT)
        wf.writeframes(pcm16.tobytes())


def synthesize_piper(voice, text: str) -> tuple[bytes, int]:
    """Returns (pcm_bytes, sample_rate) for given text."""
    chunks = list(voice.synthesize(text))
    if not chunks:
        raise RuntimeError('No audio chunks returned')
    sr    = chunks[0].sample_rate
    audio = np.concatenate([c.audio_float_array for c in chunks])
    pcm16 = np.clip(audio, -1, 1)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    return pcm16.tobytes(), sr


def main():
    from piper import PiperVoice

    sentences = [s.strip() for s in SENT_F.read_text('utf-8').splitlines() if s.strip()]
    assert len(sentences) >= 1

    for voice_dir, voice_id, gender in VOICES:
        vdir = OUT_DIR / voice_dir
        vdir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== {voice_dir} ({voice_id}, {gender}) ===')

        onnx, cfg = download_voice(voice_id)
        print(f'  Loading {onnx.name}…')
        voice = PiperVoice.load(str(onnx), str(cfg), use_cuda=False)
        sr    = voice.config.sample_rate

        for i, sent in enumerate(sentences):
            out = vdir / f'{i:03d}.wav'
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                print(f'  [{i+1:02d}/50] skip (exists)')
                continue
            print(f'  [{i+1:02d}/50] {sent[:60]}…', end=' ', flush=True)
            try:
                pcm, src_sr = synthesize_piper(voice, sent)
                save_wav_pcm(pcm, out, src_sr)
                print('ok')
            except Exception as e:
                print(f'ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
