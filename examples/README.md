# Examples

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
