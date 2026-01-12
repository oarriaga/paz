CYAN = "96"
GREEN = "92"
RED = "91"
YELLOW = "93"
GRAY = "90"
RESET = "0"


def print_colored(*args, color):
    message = " ".join(str(arg) for arg in args)
    print(f"\033[{color}m{message}\033[{RESET}m")


def info(*args):
    print_colored(*args, color=CYAN)


def success(*args):
    print_colored(*args, color=GREEN)


def error(*args):
    print_colored(*args, color=RED)


def warn(*args):
    print_colored(*args, color=YELLOW)


def debug(*args):
    print_colored(*args, color=GRAY)
