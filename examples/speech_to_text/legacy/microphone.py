import sys


def load_sounddevice():
    import sounddevice as sd

    return sd


def collect_input_devices(devices=None):
    if devices is None:
        devices = load_sounddevice().query_devices()
    input_devices = []
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            input_devices.append((index, device))
    return input_devices


def build_input_device_label(index, device):
    return "[{}] {} (inputs={}, default_sr={})".format(
        index,
        device.get("name", "unknown"),
        device.get("max_input_channels", 0),
        device.get("default_samplerate", "unknown"),
    )


def list_input_devices(stream=None, devices=None):
    if stream is None:
        stream = sys.stdout
    input_devices = collect_input_devices(devices)
    if not input_devices:
        stream.write("No input devices found.\n")
        return
    for index, device in input_devices:
        stream.write("{}\n".format(build_input_device_label(index, device)))


def resolve_input_device(selection, devices=None, default_device=None):
    if selection is None or str(selection).strip() == "":
        if default_device is None:
            default_device = load_sounddevice().query_devices(None, "input")
        return None, default_device
    if devices is None:
        devices = load_sounddevice().query_devices()
    raw_selection = str(selection).strip()
    if raw_selection.isdigit():
        device_index = int(raw_selection)
        if device_index < 0 or device_index >= len(devices):
            raise ValueError(
                "Unknown input device index: {}".format(raw_selection)
            )
        device = devices[device_index]
        if device.get("max_input_channels", 0) <= 0:
            raise ValueError(
                "Device {} is not an input device: {}".format(
                    raw_selection, device.get("name", "unknown")
                )
            )
        return device_index, device
    selection = raw_selection.lower()
    matches = []
    for index, device in collect_input_devices(devices):
        if selection in device.get("name", "").lower():
            matches.append((index, device))
    if not matches:
        raise ValueError(
            "No input device matches '{}'".format(raw_selection)
        )
    if len(matches) > 1:
        labels = ", ".join(
            build_input_device_label(index, device)
            for index, device in matches
        )
        raise ValueError(
            "Multiple input devices match '{}': {}".format(
                raw_selection, labels
            )
        )
    return matches[0]
