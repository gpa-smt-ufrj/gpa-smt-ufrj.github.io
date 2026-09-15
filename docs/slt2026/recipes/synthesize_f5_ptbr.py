"""
Synthesize the 50 pt-BR sentences using F5-TTS fine-tuned for Brazilian Portuguese.
Model: firstpixel/F5-TTS-pt-br

Two voices (male/female) by cloning azure reference clips we already have on disk.

Output:
  synth_wavs/f5_ptbr_male/000..049.wav
  synth_wavs/f5_ptbr_female/000..049.wav

Usage:
    python synthesize_f5_ptbr.py
"""
import re, struct, wave, sys
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR  = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SENT_F   = Path(os.environ.get('SYNTH_SENT_F', str(BASE / 'sentences_50.txt')))
SR_OUT   = 16000

AZURE_DIR = BASE / 'synth_wavs'

# Reference clips: azure wavs 0+1 concatenated to ~8s per gender
REFS = {
    'f5_ptbr_male': {
        'sources': ['azure_antonio/000.wav', 'azure_antonio/001.wav'],
        'text':    ('No total, sete mísseis foram disparados contra o encrave. '
                    'Até agora, na televisão, eles já somam quatro ministros.'),
    },
    'f5_ptbr_female': {
        'sources': ['azure_francisca/000.wav', 'azure_francisca/001.wav'],
        'text':    ('No total, sete mísseis foram disparados contra o encrave. '
                    'Até agora, na televisão, eles já somam quatro ministros.'),
    },
}


def load_pcm_16k(path):
    wav, sr = torchaudio.load(str(path))
    if sr != SR_OUT:
        wav = torchaudio.functional.resample(wav, sr, SR_OUT)
    return wav.mean(0)          # mono


def concat_refs(sources):
    clips = [load_pcm_16k(AZURE_DIR / s) for s in sources]
    silence = torch.zeros(int(0.2 * SR_OUT))
    out = []
    for c in clips:
        out.append(c)
        out.append(silence)
    return torch.cat(out)


def save_wav(pcm_float: np.ndarray, path: Path, sr: int = SR_OUT):
    pcm16 = np.clip(pcm_float, -1, 1)
    pcm16 = (pcm16 * 32767).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def main():
    sentences = [s.strip() for s in SENT_F.read_text('utf-8').splitlines() if s.strip()]
    assert len(sentences) >= 1

    print('Loading F5-TTS pt-BR model (firstpixel/F5-TTS-pt-br)...')
    from f5_tts.api import F5TTS
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download('firstpixel/F5-TTS-pt-br', 'pt-br/model_last.safetensors')
    model = F5TTS(
        ckpt_file=ckpt_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print('Model loaded.')

    for voice_dir, cfg in REFS.items():
        vdir = OUT_DIR / voice_dir
        vdir.mkdir(parents=True, exist_ok=True)

        # Build reference wav on disk (temp file)
        ref_pcm = concat_refs(cfg['sources'])
        tmp_ref = vdir / '_ref.wav'
        save_wav(ref_pcm.numpy(), tmp_ref)

        print(f'\n=== {voice_dir} ({ref_pcm.shape[0]/SR_OUT:.1f}s reference) ===')
        for i, sent in enumerate(sentences):
            out = vdir / f'{i:03d}.wav'
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                print(f'  [{i+1:02d}/50] skip (exists)')
                continue

            print(f'  [{i+1:02d}/50] {sent[:60]}…', end=' ', flush=True)
            try:
                wav_np, sr, _ = model.infer(
                    ref_file=str(tmp_ref),
                    ref_text=cfg['text'],
                    gen_text=sent,
                    seed=42,
                    remove_silence=True,
                )
                if sr != SR_OUT:
                    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
                    wav_t = torchaudio.functional.resample(wav_t, sr, SR_OUT)
                    wav_np = wav_t.squeeze(0).numpy()
                save_wav(wav_np, out)
                print('ok')
            except Exception as e:
                print(f'ERROR: {e}')
        tmp_ref.unlink(missing_ok=True)

    print('\nDone.')


if __name__ == '__main__':
    main()
