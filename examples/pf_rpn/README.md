# PF-RPN

Class-agnostic object proposals with PF-RPN. Every image is resized to
the model's fixed input, so the model compiles once.

## Run

```bash
python3 propose.py
python3 propose.py --images first.jpg second.jpg
```

Options: `--images` (paths; unset uses the demo images), `--size`
(two integers, default `640 640`), `--IOU_thresh` (0.5), `--top_k` (30).

With no arguments it downloads the pretrained weights and three COCO
demo images into the current directory.

## Output

For each image it prints the image size, the number of proposals and the
time taken, and writes `proposals_<index>.png` with the proposals drawn.
It ends by printing the number of compilations.
