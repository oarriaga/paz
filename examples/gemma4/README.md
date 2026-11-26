# Gemma 4

Terminal chat demos for the Gemma 4 models in
`paz.models.foundation.gemma4`.

| Script | What it does |
| --- | --- |
| `demo.py` | Text chat with `GenerateGemma4` (default `--model_name gemma4_2b`). |
| `demo_12b.py` | Text chat with `GenerateGemma412B`. |
| `demo_image.py` | Chat about an image with `DescribeImageGemma4`; the vision encoder runs on CPU. |

## Run

```bash
python3 examples/gemma4/demo.py --model_name gemma4_2b --max_tokens 256
python3 examples/gemma4/demo_image.py --image path/to/image.jpg
python3 examples/gemma4/demo_12b.py --max_tokens 64 --max_prompt 128 \
    --max_seq 256
```

Each script reads prompts from the terminal and prints the generated
text. Without `--image`, `demo_image.py` downloads a sample image.

## Requirements

- Model weights are downloaded by default; pass `--models_path` to use a
  local directory instead.
- `demo_12b.py` loads about 24 GB of weights.
