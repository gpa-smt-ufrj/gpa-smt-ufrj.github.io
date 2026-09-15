Synthesis and download recipes
==============================
Companion material for "Synthetic speech detection in Brazilian Portuguese through
accent-related features" (IEEE SLT 2026).

These are the scripts that actually produced the synthetic side of the curated dataset:
57 TTS voices from 8 providers, each reading the same 50 phonetically balanced sentences.


1. Natural speech
-----------------
The natural corpora are public. Download helpers live in the pt-br-accent-toolbox package:

    pip install git+https://github.com/pedrohlopes/pt-br-accent-toolbox
    python tools/download_datasets.py          # registry of every corpus used
    python tools/download_certas_palavras.py
    python tools/download_colingpb.py
    python tools/download_gneutral.py
    python tools/download_brspeech_df.py
    python tools/download_models.py            # ZIPA and PhoneticXeus weights

Each corpus keeps its own licence; nothing is redistributed from this page.


2. Synthetic speech
-------------------
Every script writes 16 kHz mono WAV to <SYNTH_OUT_DIR>/<voice>/000.wav .. 049.wav and reads
its prompts from <SYNTH_SENT_F>, one sentence per line.

    export ACCENTS_BASE=/path/to/data        # defaults to /mnt/data/accents_data_lake
    export SYNTH_OUT_DIR=$ACCENTS_BASE/synth_wavs
    export SYNTH_SENT_F=$ACCENTS_BASE/sentences_50.txt

Commercial providers need credentials, which are never stored in the scripts:

    python synthesize_azure.py       --key <azure_subscription_key> --region eastus
    python synthesize_google.py      --key <google_cloud_api_key>
    python synthesize_openai.py      --key <openai_api_key>
    ELEVENLABS_API_KEY=<key> python synthesize_elevenlabs.py

Open-source models download their own weights:

    python synthesize_f5_ptbr.py     # firstpixel/F5-TTS-pt-br, clones two Azure reference clips
    python synthesize_qwen3tts.py    # Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
    python synthesize_piper_ptbr.py  # rhasspy Piper pt_BR voices
    python synthesize_kokoro.py      # hexgrad/Kokoro-82M, lang_code 'p'

synth_set2_master.sh runs all eight over sentences 51-100, the longer-text set used by the
within-utterance switching experiment. It sources an .env file for the API keys.


3. Prompt selection
-------------------
select_subset.py is the greedy selection that produced sentences_50.txt from the ALCaim
1000-sentence list, minimizing the L1 distance between the subset's phone-unit distribution
and the distribution of the full list.


Synthesis ran between April and June 2026. Commercial providers may have updated their
models since; the voice identifiers in each script's VOICES list are what was requested.
