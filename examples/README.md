# Examples

`metadata_first_problem.py` plans a catalogue declaring over 128 GiB of source
files without those files existing. A separate tiny generated fixture shows an
empty training cache, explicit selected-recording acquisition, interval extraction
and cache reuse. Transfers in that fixture are local copies, not network or
large-data benchmarks. Run `python examples/metadata_first_problem.py`.

`problem_loading.py` defines classification problems over already loaded
recordings using dictionaries or the JSON files in `problem_definitions/`.
It constructs imagery and response targets on the same event intervals, a
cohort-window problem, and a separately validated participant-held-out split.
All inputs are generated. Run it from an installed source checkout:

```bash
python examples/problem_loading.py
```

`classification.ipynb` generates BrainVision recordings, converts them to BIDS,
aligns explicitly mapped channel names/orders and constructs labelled windows.
An aeon distance baseline and CSP+LDA use the same participant-held-out split.
Assertions check source preservation, exact sample picks, channel errors, case
assignments and prediction alignment. The signals are generated and establish
software behavior rather than real EEG performance.

From a source checkout, install the notebook and development dependencies:

```bash
python -m pip install -e '.[binder,dev]'
jupyter nbconvert --to notebook --execute examples/classification.ipynb \
    --output classification-executed.ipynb --output-dir /tmp
```

The development extra includes PyBV for generating BrainVision fixtures; no new
core dependency is required. The example downloads no data and cleans up its
temporary recording/BIDS files. Saved notebook outputs are unnecessary for the
source example.
