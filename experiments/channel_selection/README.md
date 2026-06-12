# Channel-selection experiments

Run DetachRocket on all configured EEG datasets:

```powershell
python experiments/channel_selection/run_channel_selection.py
```

Control dataset-level concurrency with `--workers`:

```powershell
python experiments/channel_selection/run_channel_selection.py --workers 4
```

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

Summarise classification result coverage:

```powershell
python experiments/channel_selection/summarise_results.py
```

Collate channel counts, timings, and pairwise overlap:

```powershell
python experiments/channel_selection/summarise_selection.py
```
