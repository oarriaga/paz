CYAN = "96"
GREEN = "92"
RED = "91"
YELLOW = "93"
GRAY = "90"
RESET = "0"


def print_colored(message, color):
    print(f"\033[{color}m{message}\033[{RESET}m")


def info(message):
    print_colored(message, CYAN)


def success(message):
    print_colored(message, GREEN)


def error(message):
    print_colored(message, RED)


def warn(message):
    print_colored(message, YELLOW)


def debug(message):
    print_colored(message, GRAY)
