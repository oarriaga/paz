# Writing docstrings

The API reference is generated from the docstrings by `docs/build.py`. A
docstring is Markdown: the builder copies it to the page unchanged, so it
reads the same in the source code and on the website.

## Convention

- The first line is a one-sentence summary.
- Optional sections are Markdown headings: `# Arguments`, `# Returns` and
  `# Example`.
- Sections contain ordinary Markdown: lists for arguments, fenced code
  blocks for examples.
- Names and shapes are written as inline code, for example `boxes` and
  `(num_boxes, 4)`.

## Example

This is the docstring of `compute_IOUs` in `paz/backend/boxes.py`:

````python
def compute_IOUs(boxes_A, boxes_B):
    """Computes intersection over union (IOU) between `boxes_A` and `boxes_B`.

    For each box (rows `boxes_A`) it computes the IOU to all `boxes_B`.

    # Arguments
    - `boxes_A`: Array of shape `(num_boxes_A, 4)` in corner form.
    - `boxes_B`: Array of shape `(num_boxes_B, 4)` in corner form.

    # Returns
    Array of shape `(num_boxes_A, num_boxes_B)`.

    # Example
    ```python
    boxes_A = jp.array([[0.0, 0.0, 2.0, 2.0]])
    boxes_B = jp.array([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0]])
    paz.boxes.compute_IOUs(boxes_A, boxes_B)  # [[1.0, 0.33333334]]
    ```
    """
````

The generated entry is on the [boxes](api/backend/boxes.md#compute_ious)
page. `to_center_form` and `to_corner_form` in the same module follow the
convention too.

## What the builder does

For every module listed in `docs/build.py` the builder writes one page with
the public functions and classes defined in that module, in source order.
Each entry has a heading with the name, the signature, a link to the source
and the docstring. The only change made to a docstring is that its headings
are demoted two levels, so `# Arguments` nests under the heading of its
entry.

Functions and classes without a docstring are left out. The builder prints
how many members of each module are documented:

```bash
python docs/build.py
```
