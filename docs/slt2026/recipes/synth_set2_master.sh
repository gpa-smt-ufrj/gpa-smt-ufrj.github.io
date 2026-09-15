#!/usr/bin/env bash
# Synthesize the 50 NEW (set-2) Alcaim sentences for all 8 systems into an
# isolated tree: synth_wavs_set2/<voice>/000..049.wav  (== utterances 50..99).
set -u
cd ${ACCENTS_BASE:-/mnt/data/accents_data_lake}
set -a; source .env; set +a
export SYNTH_OUT_DIR="${ACCENTS_BASE:-/mnt/data/accents_data_lake}/synth_wavs_set2"
export SYNTH_SENT_F="${ACCENTS_BASE:-/mnt/data/accents_data_lake}/sentences_51_100.txt"
mkdir -p "$SYNTH_OUT_DIR" logs_set2
LOG=logs_set2

run() { echo ">>> $1"; python3 "$@" ; echo "<<< exit $? : $1"; }

case "${1:-all}" in
  openai)     run synthesize_openai.py     --key "$OPENAI_API_KEY"          > $LOG/openai.log 2>&1 ;;
  azure)      run synthesize_azure.py      --key "$AZURE_KEY1"              > $LOG/azure.log 2>&1 ;;
  google)     run synthesize_google.py     --key "$GOOGLE_API_KEY"          > $LOG/google.log 2>&1 ;;
  elevenlabs) run synthesize_elevenlabs.py                                  > $LOG/elevenlabs.log 2>&1 ;;
  f5)         run synthesize_f5_ptbr.py                                     > $LOG/f5.log 2>&1 ;;
  qwen3)      run synthesize_qwen3tts.py                                    > $LOG/qwen3.log 2>&1 ;;
  piper)      run synthesize_piper_ptbr.py                                  > $LOG/piper.log 2>&1 ;;
  kokoro)     run synthesize_kokoro.py                                      > $LOG/kokoro.log 2>&1 ;;
  *) echo "usage: $0 {openai|azure|google|elevenlabs|f5|qwen3|piper|kokoro}"; exit 1 ;;
esac
