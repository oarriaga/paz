# PAZ documentation

Documentation uses extended Markdown, as implemented by [MkDocs](http://mkdocs.org).

## Building the documentation

- Install MkDocs: `pip install mkdocs mkdocs-material `
- `pip install -e .` to make sure that Python will import your modified version of PAZ.
- From the root directory, `cd` into the `docs/` folder and run:
    - `python autogenerate_documentation.py`
    - `mkdocs serve`    # Starts a local webserver:  [localhost:8000](http://localhost:8000)
    - `mkdocs build`    # Builds a static site in `site/` directory
