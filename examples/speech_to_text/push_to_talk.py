import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import select
import signal
import sys
import termios
import tty
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paz.applications import TranscribeWhisper
from paz.models.foundation.whisper.configuration import CONFIGS
from examples.speech_to_text.microphone import build_input_device_label
from examples.speech_to_text.microphone import collect_input_devices
from examples.speech_to_text.microphone import load_sounddevice
from examples.speech_to_text.microphone import resolve_input_device
from examples.speech_to_text.microphone import verify_input_device

SAMPLE_RATE = 16000
# Until the weights are published, load them from the local example folder.
WEIGHTS_DIR = Path(__file__).with_name("whisper_models")


RESET = "\033[0m"
BLUE = "\033[94m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"


def run(args):
    sounddevice = load_sounddevice()
    if args.list_input_devices:
        return run_list_microphones(
            args.input_device, args.verify_microphones, sounddevice
        )
    if args.verify_microphone:
        return run_verify_microphone(args.input_device, sounddevice)
    app = PushToTalk(args)
    app.sounddevice = sounddevice
    app.run()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Terminal push-to-talk demo")
    model_names = list(CONFIGS.keys())
    add = parser.add_argument
    add("--model_name", default="whisper_base_en", choices=model_names)
    add("--max_tokens", default=64, type=int)
    add("--microphone", "--input_device", dest="input_device")
    add(
        "--list_microphones",
        "--list_input_devices",
        dest="list_input_devices",
        action="store_true",
    )
    add("--verify_microphone", action="store_true")
    add("--verify_microphones", action="store_true")
    return parser


class PushToTalk:
    def __init__(self, args):
        self.model_name = args.model_name
        self.max_tokens = args.max_tokens
        self.input_device = args.input_device
        self.running = True
        self.recording = False
        self.transcribing = False
        self.stream = None
        self.device_index = None
        self.device = None
        self.sample_rate = None
        self.chunks = []
        self.transcribe = None
        self.sounddevice = None

    def run(self):
        self.install_signal_handlers()
        self.print_banner()
        self.prepare_microphone()
        self.print_status("Loading {}".format(self.model_name), BLUE)
        self.transcribe = TranscribeWhisper(
            self.model_name, self.max_tokens, models_path=str(WEIGHTS_DIR))
        self.warmup_models()
        self.print_help()
        with TerminalInput():
            while self.running:
                key = read_key()
                self.handle_key(key)
        self.close_stream()

    def handle_key(self, key):
        if key is None:
            return
        if key == "q":
            if self.recording:
                self.print_status("Stopping recording before exit", YELLOW)
                self.stop_recording()
            self.running = False
            return
        if self.transcribing:
            self.print_status("Still transcribing. Please wait...", YELLOW)
            flush_input_buffer()
            return
        if key == "r":
            self.toggle_recording()
            return
        if key == "m":
            list_microphones(self.input_device, False, self.sounddevice)
            return
        if key == "v":
            self.verify_current_microphone()

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return
        self.start_recording()

    def prepare_microphone(self):
        self.device_index, self.device = verify_input_device(
            self.input_device, sounddevice=self.sounddevice
        )
        self.sample_rate = compute_sample_rate(self.device)
        label = build_device_label(self.device_index, self.device)
        self.print_status("Using microphone {}".format(label), GREEN)

    def start_recording(self):
        try:
            stream = self.build_stream(self.device_index, self.sample_rate)
        except Exception as error:
            print_error("Could not prepare recording: {}".format(error))
            return
        try:
            stream.start()
        except Exception as error:
            try:
                stream.close()
            except Exception:
                pass
            print_error("Could not start recording: {}".format(error))
            return
        self.stream = stream
        self.recording = True
        self.chunks = []
        label = build_device_label(self.device_index, self.device)
        self.print_status("Recording from {}".format(label), GREEN)
        self.print_status("Press r again to stop and transcribe", BLUE)

    def stop_recording(self):
        stream = self.stream
        sample_rate = self.sample_rate
        self.stream = None
        self.recording = False
        if stream is None:
            self.print_status("No active recording to stop", YELLOW)
            return
        self.transcribing = True
        self.print_status("Stopping recording...", YELLOW)
        try:
            stream.stop()
            stream.close()
        except Exception as error:
            self.transcribing = False
            print_error("Could not stop recording: {}".format(error))
            return
        waveform = build_waveform(self.chunks)
        self.chunks = []
        if waveform.size == 0:
            self.transcribing = False
            self.print_status("Skipping empty recording", YELLOW)
            return
        seconds = len(waveform) / float(sample_rate)
        self.print_status(
            "Finished recording {:.1f}s. Transcribing...".format(seconds),
            YELLOW,
        )
        flush_input_buffer()
        try:
            text = self.transcribe(waveform, sample_rate)
        except Exception as error:
            self.transcribing = False
            print_error("Transcription failed: {}".format(error))
            return
        self.transcribing = False
        text = text.strip()
        if is_blank_text(text):
            self.print_status("No speech detected. Try again.", YELLOW)
            return
        self.print_transcript(text)
        self.print_status("Ready for another recording", BLUE)

    def verify_current_microphone(self):
        try:
            verify_input_device(
                self.device_index, sounddevice=self.sounddevice
            )
        except Exception as error:
            print_error("Microphone check failed: {}".format(error))
            return
        label = build_device_label(self.device_index, self.device)
        self.print_status("Microphone ready: {}".format(label), GREEN)

    def warmup_models(self):
        self.print_status("Warming up on silence...", BLUE)
        waveform = np.zeros((SAMPLE_RATE,), dtype="float32")
        try:
            self.transcribe(waveform, SAMPLE_RATE)
        except Exception as error:
            raise RuntimeError("Warm-up failed: {}".format(error)) from error
        self.print_status("Warm-up complete", GREEN)

    def build_stream(self, device_index, sample_rate):
        return self.sounddevice.InputStream(
            device=device_index,
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self.collect_chunk,
        )

    def collect_chunk(self, indata, _frames, _time, _status):
        chunk = np.asarray(indata[:, 0], dtype="float32").copy()
        self.chunks.append(chunk)

    def close_stream(self):
        stream = self.stream
        self.stream = None
        self.recording = False
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def install_signal_handlers(self):
        signal.signal(signal.SIGINT, self.stop_running)
        signal.signal(signal.SIGTERM, self.stop_running)

    def stop_running(self, _signum, _frame):
        self.running = False

    def print_banner(self):
        message = "PAZ Whisper push-to-talk"
        print_color(message, CYAN)

    def print_help(self):
        message = "Keys: m list, v verify, r start/stop, q quit"
        print_color(message, BLUE)
        message = "After you start recording, press r again to stop."
        print_color(message, BLUE)

    def print_status(self, message, color):
        print_color(message, color)

    def print_transcript(self, text):
        print_color("Transcript: {}".format(text), CYAN)


class TerminalInput:
    def __enter__(self):
        ensure_terminal()
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, _type, _value, _traceback):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


def build_waveform(chunks):
    if not chunks:
        return np.array([], dtype="float32")
    return np.concatenate(chunks, axis=0).astype("float32")


def compute_sample_rate(device):
    return int(round(device.get("default_samplerate", 16000)))


def build_device_label(device_index, device):
    if device_index is None:
        name = device.get("name", "default input")
        return name
    return build_input_device_label(device_index, device)


def run_list_microphones(selection=None, verify=False, sounddevice=None):
    list_microphones(selection, verify, sounddevice)
    return 0


def run_verify_microphone(selection=None, sounddevice=None):
    sounddevice = load_sounddevice() if sounddevice is None else sounddevice
    index, device = verify_input_device(selection, sounddevice=sounddevice)
    label = build_device_label(index, device)
    print_color("Microphone ready: {}".format(label), GREEN)
    return 0


def list_microphones(selection=None, verify=False, sounddevice=None):
    sounddevice = load_sounddevice() if sounddevice is None else sounddevice
    devices = sounddevice.query_devices()
    selected_index = find_selected_index(selection, devices)
    for index, device in collect_input_devices(devices):
        line = build_microphone_line(index, device, selected_index)
        if not verify:
            print(line)
            continue
        status = verify_microphone_line(index, sounddevice)
        print("{} {}".format(line, status))


def find_selected_index(selection, devices):
    if selection is None or str(selection).strip() == "":
        return None
    index, _ = resolve_input_device(selection, devices)
    return index


def build_microphone_line(index, device, selected_index=None):
    label = build_input_device_label(index, device)
    if selected_index == index:
        return "{} [selected]".format(label)
    return label


def verify_microphone_line(index, sounddevice):
    try:
        verify_input_device(index, sounddevice=sounddevice)
    except Exception as error:
        return "[failed: {}]".format(error)
    return "[ok]"


def is_blank_text(text):
    if not text:
        return True
    return text.strip() == "[BLANK_AUDIO]"


def read_key(timeout=0.1):
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    return sys.stdin.read(1)


def ensure_terminal():
    if not sys.stdin.isatty():
        raise ValueError("This demo requires a terminal.")


def flush_input_buffer():
    if not sys.stdin.isatty():
        return
    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)


def print_color(message, color):
    print("{}{}{}".format(color, message, RESET), flush=True)


def print_error(message):
    print("{}{}{}".format(RED, message, RESET), file=sys.stderr, flush=True)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except (ImportError, ValueError) as error:
        print_error(str(error))
        raise SystemExit(1)
