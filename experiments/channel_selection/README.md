# Channel-selection experiments

Run DetachRocket on all configured EEG datasets:

```powershell
python experiments/channel_selection/run_channel_selection.py
```

Control dataset-level concurrency with `--workers`:

```powershell
python experiments/channel_selection/run_channel_selection.py --workers 4
```

For a cluster job, run exactly one algorithm and dataset with the
single-process entry point:

```bash
python experiments/channel_selection/run_channel_selection_cluster.py \
  ECP Alzheimers \
  --data-root /path/to/eeg \
  --output-root /path/to/channel-selection
```

The algorithm and dataset are positional arguments. Set `DEFAULT_DATA_ROOT` and
`DEFAULT_OUTPUT_ROOT` in the script if these paths are fixed on the cluster, in
which case the command is simply:

```bash
python experiments/channel_selection/run_channel_selection_cluster.py \
  ECP Alzheimers
```

Each cluster job writes its summary to its dataset output directory, avoiding a
shared summary file when scheduler jobs run concurrently. Use `--overwrite` to
rerun an already completed algorithm-dataset pair.

Run CSP channel creation, retaining approximately 25% as spatial components:

```powershell
python experiments/channel_selection/run_channel_selection.py --selectors CSP
```

Run binary particle swarm optimisation channel selection:

```powershell
python experiments/channel_selection/run_channel_selection.py --selectors BPSO
```

Run UMAP channel creation, retaining approximately 25% as latent channels:

```powershell
python experiments/channel_selection/run_channel_selection.py --selectors UMAP
```

Run training-case or time-axis reduction using an IndividualTDE proxy:

```powershell
python experiments/channel_selection/run_channel_selection.py \
  --selectors CaseTimeReducer
```

Case reduction is applied only to TRAIN. Any selected time reduction is applied
to both TRAIN and TEST.

Apply CaseTimeReducer after ECP and write the chained experiment to
`CaseTimeReducerECP`:

```powershell
python experiments/channel_selection/run_channel_selection.py `
  --selectors CaseTimeReducer `
  --data-root "D:\Data\ChannelSelection\ECP" `
  --output-root "D:\Data\ChannelSelection" `
  --output-name CaseTimeReducerECP
```

Summarise classification result coverage:

```powershell
python experiments/channel_selection/summarise_results.py
```

Collate channel counts, timings, and pairwise overlap:

```powershell
python experiments/channel_selection/summarise_selection.py
```
