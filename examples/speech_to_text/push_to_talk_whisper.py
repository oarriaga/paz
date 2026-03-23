import os

os.environ["KERAS_BACKEND"] = "jax"

import argparse
import logging
import selectors
import signal
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.speech_to_text.microphone import build_input_device_label
from examples.speech_to_text.microphone import list_input_devices
from examples.speech_to_text.microphone import resolve_input_device


DEFAULT_FIFO_PATH = "/tmp/whisper_ptt_fifo"
DEFAULT_MAX_TOKENS = 64


def build_parser():
    parser = argparse.ArgumentParser(description="Push-to-talk Whisper daemon")
    parser.add_argument("--fifo-path", default=DEFAULT_FIFO_PATH)
    parser.add_argument("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int)
    parser.add_argument(
        "--input-device",
        default=os.environ.get("WHISPER_PTT_INPUT_DEVICE"),
    )
    parser.add_argument(
        "--list-input-devices",
        action="store_true",
    )
    return parser

class PushToTalkWhisper:
    def __init__(self, fifo_path, max_tokens, input_device=None):
        self.fifo_path = fifo_path
        self.max_tokens = max_tokens
        self.input_device = input_device
        self.running = True
        self.recording = False
        self.transcribing = False
        self.stream = None
        self.sample_rate = None
        self.chunks = []
        self.lock = threading.Lock()
        self.command_buffer = ""
        self.selector = selectors.DefaultSelector()
        self.fifo_reader = None
        self.fifo_writer = None
        self.frontend = None
        self.encoder = None
        self.cross_cache_model = None
        self.decoder_step_model = None
        self.decoder = None
        self.preprocess_waveform = None
        self.transcribe_waveform = None

    def run(self):
        self._install_signal_handlers()
        self._load_models()
        self._open_fifo()
        logging.info("Push-to-talk daemon ready")
        try:
            while self.running:
                events = self.selector.select(timeout=0.5)
                for key, _ in events:
                    self._read_commands(key.fd)
        finally:
            self._close_stream()
            self._close_fifo()

    def _install_signal_handlers(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, _frame):
        logging.info("Stopping daemon on signal %s", signum)
        self.running = False

    def _load_models(self):
        from examples.speech_to_text.decoding import (
            KVDecoder,
            build_whisper_base_en_prompt_token_ids,
        )
        from examples.speech_to_text.demo import (
            preprocess_waveform,
            transcribe_waveform,
        )
        from examples.speech_to_text.model2 import (
            CONFIGS,
            WhisperCrossCache,
            WhisperDecoderStep,
            WhisperEncoder,
            WhisperFrontend,
        )

        config = CONFIGS["whisper_base_en"]
        logging.info("Loading Whisper models")
        self.preprocess_waveform = preprocess_waveform
        self.transcribe_waveform = transcribe_waveform
        self.frontend = WhisperFrontend()
        self.encoder = WhisperEncoder(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_encoder",
        )
        self.cross_cache_model = WhisperCrossCache(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_cross_cache",
        )
        self.decoder_step_model = WhisperDecoderStep(
            **config,
            weights="whisper_base_en",
            name="whisper_base_en_decoder_step",
        )
        prompt_token_ids = build_whisper_base_en_prompt_token_ids()
        self.decoder = KVDecoder(
            self.decoder_step_model, prompt_token_ids, self.max_tokens
        )

    def _open_fifo(self):
        fifo_path = Path(self.fifo_path)
        if fifo_path.exists() and not fifo_path.is_fifo():
            raise ValueError("{} exists and is not a FIFO".format(fifo_path))
        if not fifo_path.exists():
            os.mkfifo(fifo_path)
        self.fifo_reader = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        self.fifo_writer = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        self.selector.register(self.fifo_reader, selectors.EVENT_READ)

    def _close_fifo(self):
        if self.fifo_reader is not None:
            try:
                self.selector.unregister(self.fifo_reader)
            except Exception:
                pass
            os.close(self.fifo_reader)
            self.fifo_reader = None
        if self.fifo_writer is not None:
            os.close(self.fifo_writer)
            self.fifo_writer = None
        fifo_path = Path(self.fifo_path)
        if fifo_path.exists() and fifo_path.is_fifo():
            fifo_path.unlink()

    def _read_commands(self, fd):
        try:
            payload = os.read(fd, 4096)
        except BlockingIOError:
            return
        if not payload:
            return
        self.command_buffer += payload.decode("utf-8", errors="ignore")
        while "\n" in self.command_buffer:
            command, self.command_buffer = self.command_buffer.split("\n", 1)
            self._handle_command(command.strip())

    def _handle_command(self, command):
        if not command:
            return
        if command == "start":
            self._start_recording()
            return
        if command == "stop":
            self._stop_recording()
            return
        logging.error("Unknown command: %s", command)

    def _resolve_input_device(self):
        return resolve_input_device(self.input_device)

    def _audio_callback(self, indata, _frames, _time, status):
        del status
        with self.lock:
            self.chunks.append(indata.copy())

    def _start_recording(self):
        with self.lock:
            if self.recording:
                logging.info("Ignoring duplicate start while recording")
                return
            if self.transcribing:
                logging.info("Ignoring start while transcribing")
                return
            try:
                input_device, device_info = self._resolve_input_device()
            except ValueError as error:
                logging.error("%s", error)
                return
            self.sample_rate = int(
                round(device_info.get("default_samplerate", 16000))
            )
            self.chunks = []
        stream = None
        try:
            stream = sd.InputStream(
                device=input_device,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            stream.start()
        except Exception:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            logging.exception("Failed to start recording")
            return
        with self.lock:
            self.stream = stream
            self.recording = True
        logging.info(
            "Recording started at %s Hz from %s",
            self.sample_rate,
            build_input_device_label(input_device, device_info)
            if input_device is not None
            else device_info.get("name", "default input"),
        )

    def _stop_recording(self):
        with self.lock:
            if not self.recording:
                logging.info("Ignoring stop while idle")
                return
            self.recording = False
            stream = self.stream
            self.stream = None
            sample_rate = self.sample_rate
            self.sample_rate = None
        should_transcribe = True
        try:
            if stream is not None:
                stream.stop()
                stream.close()
        finally:
            with self.lock:
                chunks = self.chunks
                self.chunks = []
                if self.transcribing:
                    logging.info("Ignoring stop while transcribing")
                    should_transcribe = False
                else:
                    self.transcribing = True
        if not should_transcribe:
            return
        waveform = self._build_waveform(chunks)
        worker = threading.Thread(
            target=self._transcribe_and_type,
            args=(waveform, sample_rate),
            daemon=True,
        )
        worker.start()

    def _close_stream(self):
        with self.lock:
            stream = self.stream
            self.stream = None
            self.recording = False
        if stream is not None:
            try:
                stream.stop()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    def _build_waveform(self, chunks):
        if not chunks:
            return np.array([], dtype="float32")
        waveform = np.concatenate(chunks, axis=0)
        return np.asarray(waveform, dtype="float32")

    def _transcribe_and_type(self, waveform, sample_rate):
        try:
            if sample_rate is None or waveform.size == 0:
                logging.info("Skipping empty recording")
                return
            _, waveform = self.preprocess_waveform(waveform, sample_rate)
            _, _, decoded_text = self.transcribe_waveform(
                waveform,
                frontend=self.frontend,
                encoder=self.encoder,
                cross_cache_model=self.cross_cache_model,
                decoder_step_model=self.decoder_step_model,
                decoder=self.decoder,
                max_tokens=self.max_tokens,
            )
            decoded_text = decoded_text.strip()
            if not decoded_text:
                logging.info("Skipping empty transcription")
                return
            self._type_text(decoded_text)
            logging.info("Typed transcription: %s", decoded_text)
        except Exception:
            logging.exception("Transcription failed")
        finally:
            with self.lock:
                self.transcribing = False

    def _type_text(self, text):
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "0", "--", text],
            check=True,
        )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args = build_parser().parse_args()
    if args.list_input_devices:
        list_input_devices()
        return
    daemon = PushToTalkWhisper(
        args.fifo_path, args.max_tokens, args.input_device
    )
    daemon.run()


if __name__ == "__main__":
    main()
