"""Chats with the Gemma 4 served by `serve.py` on another computer.

Open a tunnel to the server, then run the demo with the server's secret:

    ssh -N -L 5000:localhost:5000 <server>
    PAZ_REMOTE_KEY=<secret> python demo.py
"""
import argparse
import os

import paz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    authkey = os.environ["PAZ_REMOTE_KEY"].encode()
    connection = paz.remote.connect(args.port, authkey)
    print("Type a prompt and press Enter. Submit an empty line to quit.")
    prompt = input("you> ").strip()
    while prompt != "":
        print("gemma> ", end="", flush=True)
        for piece in paz.remote.stream(connection, "chat", prompt):
            print(piece, end="", flush=True)
        print()
        prompt = input("you> ").strip()
