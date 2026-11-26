# Remote functions

Serve a function in one process with `paz.remote.serve` and call it from
another with `paz.remote.connect`, `paz.remote.call` and
`paz.remote.stream`.

| Script | What it does |
| --- | --- |
| `chat/serve.py` | Serves a Gemma 4 `chat` function that streams generated text. |
| `chat/demo.py` | Reads prompts from the terminal and prints the streamed reply. |
| `detection/serve.py` | Serves a `detect` function. |
| `detection/demo.py` | Sends camera frames to `detect` and uses the returned result. |

## Run

Start the server, then the client in a second terminal:

```bash
python3 examples/remote/chat/serve.py --port 5000 --model_name gemma4_2b
python3 examples/remote/chat/demo.py --port 5000
```

The client prints `Type a prompt and press Enter. Submit an empty line to
quit.` and streams each reply after `gemma> `.

Other options: `chat/serve.py` takes `--max_tokens`, `--max_seq` and
`--max_prompt`; `detection/serve.py` takes `--port` and `--score_thresh`;
`detection/demo.py` takes `--port`, `--camera`, `--H`, `--W` and
`--quality`.

## Requirements

- The chat server loads the Gemma 4 weights.
- `detection/demo.py` needs a camera.
