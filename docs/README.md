# Building the documentation

From the repository root, install the documentation dependencies in a virtual
environment and build the HTML pages:

```bash
python -m pip install ".[docs]"
python -m sphinx -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to inspect the result. The build includes the
API reference and linked Markdown guides. Example notebooks are not executed.

Read the Docs uses Ubuntu 24.04 and Python 3.12, configured in
`.readthedocs.yml`, and installs the same `docs` extra. The extra includes
`sphinx-remove-toctrees` because `docs/conf.py` enables that extension in addition
to the extensions supplied by `aeon[docs]`.

The hosted `latest` documentation follows `main`; `stable` follows the selected
release tag. Changes merged into `main` appear on `latest` after a successful
Read the Docs build. They do not update an older release's documentation.
If no build is triggered after a merge, a Read the Docs project maintainer
should check the GitHub integration and trigger a build from the dashboard.
