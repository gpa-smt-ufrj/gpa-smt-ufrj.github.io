"""
Synthesize the 50 pt-BR sentences using Kokoro-82M (pt-BR voices).

Three pt-BR voices:
  pf_dora  — female
  pm_alex  — male
  pm_santa — male

Output:
  synth_wavs/kokoro_pf_dora/000..049.wav
  synth_wavs/kokoro_pm_alex/000..049.wav
  synth_wavs/kokoro_pm_santa/000..049.wav

Usage:
    python synthesize_kokoro.py
"""
import numpy as np
import soundfile as sf
import torch
import os
from pathlib import Path

BASE = Path(os.environ.get('ACCENTS_BASE', '/mnt/data/accents_data_lake'))
OUT_DIR = Path(os.environ.get('SYNTH_OUT_DIR', str(BASE / 'synth_wavs')))
SENT_F  = Path(os.environ.get('SYNTH_SENT_F', str(BASE / 'sentences_50.txt')))
SR_OUT  = 16000

VOICES = [
    ('kokoro_pf_dora',  'pf_dora'),
    ('kokoro_pm_alex',  'pm_alex'),
    ('kokoro_pm_santa', 'pm_santa'),
]


def resample_if_needed(audio: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == SR_OUT:
        return audio
    import torchaudio
    t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    t = torchaudio.functional.resample(t, src_sr, SR_OUT)
    return t.squeeze(0).numpy()


def main():
    sentences = [s.strip() for s in SENT_F.read_text('utf-8').splitlines() if s.strip()]
    assert len(sentences) >= 1

    from kokoro import KPipeline
    print('Loading Kokoro-82M (pt-BR)...')
    pipe = KPipeline(lang_code='p', repo_id='hexgrad/Kokoro-82M')
    print('Pipeline ready.')

    for voice_dir, voice_id in VOICES:
        vdir = OUT_DIR / voice_dir
        vdir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== {voice_dir} (voice={voice_id}) ===')

        for i, sent in enumerate(sentences):
            out = vdir / f'{i:03d}.wav'
            if out.exists():  # EXISTS-SKIP
                continue
            if out.exists():
                print(f'  [{i+1:02d}/50] skip (exists)')
                continue

            print(f'  [{i+1:02d}/50] {sent[:60]}...', end=' ', flush=True)
            try:
                audio_chunks = []
                for result in pipe(sent, voice=voice_id):
                    model_out = result.output
                    if model_out is None:
                        continue
                    audio = model_out.audio if hasattr(model_out, 'audio') else model_out
                    if audio is None:
                        continue
                    arr = audio.numpy() if hasattr(audio, 'numpy') else np.array(audio)
                    if arr.ndim == 0:
                        continue
                    audio_chunks.append(arr.squeeze())

                if not audio_chunks:
                    print('ERROR: no audio generated')
                    continue

                audio = np.concatenate(audio_chunks)
                audio = resample_if_needed(audio.astype(np.float32), 24000)
                sf.write(str(out), audio, SR_OUT)
                print(f'ok ({len(audio)/SR_OUT:.1f}s)')
            except Exception as e:
                print(f'ERROR: {e}')

    print('\nDone.')


if __name__ == '__main__':
    main()
