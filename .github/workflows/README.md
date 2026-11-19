# Workflow runner images

The PR, periodic and release test matrices use Ubuntu 24.04, macOS 15 and
Windows 2022. Keep their pinned OS labels consistent when migrating images.
The standard `macos-15` label uses GitHub's Arm64 image.

macOS 14 jobs are deliberately cancelled during GitHub's retirement brownouts;
that image retires on 2 November 2026. A rerun cannot resolve a scheduled brownout.
Use a supported pinned image rather than `macos-latest` so that an OS migration
remains an explicit change.

See the [retirement notice](https://github.com/actions/runner-images/issues/13518)
and [current runner labels](https://github.com/actions/runner-images#available-images).

## Optional Zenodo download tests

The default suite checks EEG dataset loading with local files and a mocked
download. To run the live Sleep download smoke test, enable the download tests
and select that test:

```bash
AEON_NEURO_DOWNLOAD_TESTS=1 python -m pytest \
  aeon_neuro/datasets/tests/test_data_loaders.py::test_load_eeg_from_zenodo
```

Run the whole test module with the same flag to check downloads for all
available EEG datasets.
