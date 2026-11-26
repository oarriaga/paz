# Speech to text

Transcription with the Whisper models through `TranscribeWhisper`.

| Script | What it does |
| --- | --- |
| `demo.py` | Loads an audio file with `paz.audio.load` and prints its transcription. |
| `push_to_talk.py` | Terminal push-to-talk demo that records from a microphone and transcribes. |
| `microphone.py` | Microphone listing and verification helpers built on `sounddevice`. |
| `push_to_talk_test.py` | Tests for the microphone helpers and the push-to-talk demo. |

## Run

```bash
python3 examples/speech_to_text/demo.py --audio_path speech.wav \
    --model_name whisper_base_en --max_tokens 64
python3 -m examples.speech_to_text.push_to_talk --list_microphones
python3 -m examples.speech_to_text.push_to_talk --microphone 0
```

`demo.py` prints the transcribed text. `push_to_talk.py` also accepts
`--verify_microphone` and `--verify_microphones`; run it from the
repository root because it imports `examples.speech_to_text.microphone`.

## Requirements

- `--audio_path` defaults to `harvard.wav` in this folder, which is not
  shipped; pass your own audio file.
- `push_to_talk.py` needs a microphone, `pip install sounddevice` and
  PortAudio.
