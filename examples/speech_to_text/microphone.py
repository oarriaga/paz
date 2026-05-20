import importlib
import sys


def load_sounddevice():
    try:
        return importlib.import_module("sounddevice")
    except (ImportError, OSError) as error:
        raise ImportError(build_sounddevice_error()) from error


def build_sounddevice_error():
    return build_sounddevice_messages()


def build_sounddevice_messages():
    messages = [
        "Microphone support requires sounddevice and PortAudio.",
        "Python package: pip install sounddevice",
        "System package: install PortAudio for your OS.",
    ]
    return " ".join(messages)


def collect_input_devices(devices=None):
    if devices is None:
        devices = load_sounddevice().query_devices()
    input_devices = []
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            input_devices.append((index, device))
    return input_devices


def build_input_device_label(index, device):
    values = (
        index,
        device.get("name", "unknown"),
        device.get("max_input_channels", 0),
        device.get("default_samplerate", "unknown"),
    )
    return "[{}] {} (inputs={}, default_sr={})".format(*values)


def list_input_devices(stream=None, devices=None):
    stream = sys.stdout if stream is None else stream
    input_devices = collect_input_devices(devices)
    if not input_devices:
        stream.write("No input devices found.\n")
        return
    for index, device in input_devices:
        stream.write("{}\n".format(build_input_device_label(index, device)))


def resolve_input_device(
    selection, devices=None, default_device=None, sounddevice=None
):
    if selection is None or str(selection).strip() == "":
        if default_device is None:
            sounddevice = (
                load_sounddevice() if sounddevice is None else sounddevice
            )
            default_device = sounddevice.query_devices(None, "input")
        return None, default_device
    if devices is None:
        sounddevice = (
            load_sounddevice() if sounddevice is None else sounddevice
        )
        devices = sounddevice.query_devices()
    selection = str(selection).strip()
    if selection.isdigit():
        return resolve_device_index(int(selection), devices)
    return resolve_device_name(selection, devices)


def resolve_device_index(device_index, devices):
    if device_index < 0 or device_index >= len(devices):
        raise ValueError(
            "Unknown input device index: {}".format(device_index)
        )
    device = devices[device_index]
    if device.get("max_input_channels", 0) <= 0:
        name = device.get("name", "unknown")
        raise ValueError(
            "Device {} is not an input device: {}".format(
                device_index, name
            )
        )
    return device_index, device


def resolve_device_name(selection, devices):
    lowered = selection.lower()
    matches = []
    for index, device in collect_input_devices(devices):
        if lowered in device.get("name", "").lower():
            matches.append((index, device))
    if not matches:
        raise ValueError(
            "No input device matches '{}'".format(selection)
        )
    if len(matches) > 1:
        labels = []
        for index, device in matches:
            labels.append(build_input_device_label(index, device))
        raise ValueError(
            "Multiple input devices match '{}': {}".format(
                selection, ", ".join(labels)
            )
        )
    return matches[0]


def verify_input_device(selection, devices=None, sounddevice=None):
    sounddevice = load_sounddevice() if sounddevice is None else sounddevice
    if devices is None:
        devices = sounddevice.query_devices()
    default_device = sounddevice.query_devices(None, "input")
    args = (selection, devices, default_device, sounddevice)
    index, device = resolve_input_device(*args)
    sample_rate = int(round(device.get("default_samplerate", 16000)))
    kwargs = {
        "device": index,
        "samplerate": sample_rate,
        "channels": 1,
        "dtype": "float32",
    }
    sounddevice.check_input_settings(**kwargs)
    stream = sounddevice.InputStream(**kwargs)
    try:
        stream.start()
        sounddevice.sleep(50)
    finally:
        stream.stop()
        stream.close()
    return index, device
