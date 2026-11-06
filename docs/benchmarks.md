# EEG benchmarks

A survey of EEG decoding benchmarks, oldest first, compiled November 2026.

## Competitions (fixed datasets, one-off results)

| Released | Benchmark | Scope | Results |
|---|---|---|---|
| 2008 | BCI Competition IV | Motor imagery and others, 5 datasets. Dataset 2a is MOABB's `BNCI2014_001`. | Winners list only |
| 2021 | BEETL (NeurIPS 2021 competition) | Transfer learning for sleep staging and motor imagery | Competition leaderboard |
| 2025 | NeurIPS 2025 EEG Foundation Challenge | Cross-task and cross-subject decoding on HBN data. Organised by people from the braindecode group. | Competition leaderboard |

## Benchmarks of classic models

Classic deep learning models, plus non-deep-learning baselines.

| Released | Benchmark | Scope | Models | Results table? |
|---|---|---|---|---|
| 2018, benchmark 2024 | [MOABB](https://moabb.neurotechx.com/docs/paper_results.html): framework by Jayaram and Barachant (2018); benchmark by [Chevallier et al. (arXiv, April 2024)](https://arxiv.org/abs/2404.15319) | 36 datasets in the benchmark (motor imagery, P300, SSVEP), from 206 in the library | 30 pipelines: 11 raw signal, 13 Riemannian, and 6 deep learning (ShallowConvNet, DeepConvNet, EEGNet, EEGTCNet, EEGITNet, EEGNeX) | Yes, a public table: within-session 5-fold CV, ROC-AUC (2 classes) or accuracy |
| 2018 | [Heilmeyer et al.](https://arxiv.org/pdf/1806.07741) | Framework for large-scale evaluation | Deep4, Shallow, EEGNet v1 and v2 | Paper only |
| 2022, updated 2023 | [EEGNeX paper](https://arxiv.org/abs/2207.12369) | 11 MOABB motor imagery datasets | EEGNeX against EEGNet and others | Paper only; the authors evaluated their own model |
| about 2022 | [Riemann vs CNN](https://eprints.cs.univie.ac.at/8125) (Univie) | MOABB motor imagery; within-session, cross-session and cross-subject | EEGNet, Shallow, Deep against Riemannian methods | Paper only |
| 2026 | [Low-cost motor imagery study](https://neurips.cc/virtual/2026/158006) (NeurIPS 2026 listing) | Motor imagery on cheap headsets | CSP+LDA, EEGNet, ShallowConvNet, DeepConvNet, EEGConformer | Paper only |

## Benchmarks of foundation models

Large pretrained EEG models.

| Released | Benchmark | Scope | Models | Results table? |
|---|---|---|---|---|
| Jul 2025 | [AdaBrain-Bench](https://arxiv.org/abs/2507.09882) (Wu et al.) | BCI datasets across 7 application types; cross-subject, multi-subject and few-shot | Public foundation models such as LaBraM, CBraMod and EEGPT, with EEGNet as a baseline | Paper only |
| Aug 2025 | [EEG-FM-Bench](https://arxiv.org/abs/2508.17742) (Xiong et al.) | 14 datasets, 10 paradigms (motor imagery, sleep, emotion, disease) | BIOT, BENDR, LaBraM, EEGPT, CBraMod, CSBrain, REVE, plus the general time-series models Mantis and Moment | Paper only; code at [xw1216/EEG-FM-Bench](https://github.com/xw1216/EEG-FM-Bench) |
| Nov 2025 | [EEG-Bench](https://alphaxiv.org/abs/2512.08959) (Kastrati et al., ETH Zurich) | Clinical: 11 diagnostic tasks, 14 datasets (epilepsy, schizophrenia, Parkinson's, OCD, mild traumatic brain injury) | Foundation models and classical baselines | Paper; data and code released |
| 2026 | [OpenEEG-Bench](https://github.com/braindecode/OpenEEGBench) (braindecode group, [EUSIPCO 2026](https://sccn.ucsd.edu/2026_EUSIPCO_benchmark.pdf)) | About 10 datasets: PhysioNet motor imagery, CHB-MIT, FACED, ISRUC-Sleep, TUAB, TUEV and others | Fine-tuning strategies for foundation models | A live [Hugging Face leaderboard](https://huggingface.co/spaces/braindecode/OpenEEGBench) |
| May 2026 | [NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/index.html) ([Banville et al., Meta, arXiv 2605.08495](https://arxiv.org/abs/2605.08495)) | NeuralBench-EEG v1.0: 36 tasks (BCI, ERP, clinical, sleep, decoding, regression). Core suite has one dataset per task; Full suite has 94 datasets. Preliminary MEG, fMRI and EMG tasks. | 8 task-specific models (ShallowFBCSPNet, Deep4Net, EEGNet, BDTCN, ATCNet, EEGConformer, CTNet, SimpleConvTimeAgg); 6 foundation models (BENDR, BIOT, LaBraM, CBraMod, LUNA, REVE); Riemannian baselines (CovTsLR, XdawnTsLR, CoSpectraLogLR, CovTsRidge); dummy and chance | Interactive leaderboard; per-task scores only in figures. Code at [facebookresearch/neuroai](https://github.com/facebookresearch/neuroai), `pip install neuralbench`. |
| May 2026 | [OmniEEG-Bench](https://arxiv.org/abs/2606.00815) (Lu et al.) | 54 datasets, 58 tasks in 6 families | 10 foundation models | Paper; code at ncclab-sustech/omni-eegbench |
| Sep 2026 | [EEG-Arena](https://eeg.fm/benchmarks/) ([Chen et al., arXiv 2609.32743](https://arxiv.org/abs/2609.32743); PolyU MIND Lab and XJTLU CICO) | 57 tasks from 23 public datasets, 7 paradigms (the paper). The toolkit now lists 29 datasets and 74 tasks. Five protocols: overall performance, pretraining effectiveness, model scaling, pretraining data scaling, channel configuration. | 30 foundation models (variants of BENDR, BIOT, EEGPT, LaBraM, NeuroGPT, CBraMod, CSBrain, EEGMamba, HEAR, LEAD, LUNA, mdJPT, NeuroRVQ, REVE, CodeBrain, ST-EEGFormer, SleepFM, SleepGPT, TFM-Tokenizer, Uni-DMFM). 25 supervised baselines, all deep except a YASA-style sleep staging baseline: EEGNet, ShallowConvNet, DeepConvNet, EEGConformer, ATCNet, CTNet, MSVTNet, EEGInceptionERP, DBConformer, FBCNet, IFNet, EEGWaveNet, SlimSeiz, MSCFormer, TMSA-Net, ADFCNN, EEGNeX, BiDANN, TSception, HSLT, DeepSleepNet, SleepEEGNet, SleepTransformer, U-Sleep. | Public [leaderboard](https://eeg.fm/leaderboards/eeg-arena/) and [results site](https://eegarena.eeg.fm/), both loaded by JavaScript, so not readable here. Code at [OrangeP0P/eeg-arena](https://github.com/OrangeP0P/eeg-arena) (toolkit v0.00, October 2026, no licence file found); standardised data on [ModelScope](https://www.modelscope.cn/datasets/orangeeg/eeg-arena). |

## Findings repeated across the benchmarks

- Riemannian methods match or beat classic deep learning on standard BCI data
  (MOABB, Univie).
- Simple models often stay competitive with foundation models, especially under
  clinical distribution shift (EEG-Bench).
- Frozen foundation-model features transfer poorly (EEG-FM-Bench).
- Foundation models do better with more diverse pretraining data and more
  parameters (OmniEEG-Bench).
- Foundation models "only marginally outperform task-specific models", and
  clinical and cognitive decoding tasks remain hard for every model
  (NeuralBench). CTNet, with about 150K parameters, is close behind REVE, with
  69M.

- Foundation models beat strong supervised baselines on most tasks, especially
  with non-bipolar montages. Bigger foundation models are not consistently
  better, but more pretraining data helps (EEG-Arena). This is a more positive
  view of foundation models than NeuralBench's.

## Pretraining leakage

Foundation models can be pretrained on the same datasets, or even the same
subjects, as the downstream test sets. The benchmarks handle this differently:

- **EEG-Arena** marks tasks whose data overlap a model's pretraining data
  (red "E" badges) and excludes them from that model's ranking.
- **NeuralBench** flags overlap (hatched bars in its figures) but keeps those
  results. It argues there is "no clear trend" of leakage helping, while
  admitting the scores "could be overinflated".
- **The other foundation-model benchmarks:** we haven't checked how they
  handle it.

Any comparison with foundation models should check each model's pretraining
corpus against the test datasets. Models trained from scratch avoid the
problem.

## NeuralBench in more detail

NeuralBench is the most useful benchmark here for aeon-neuro, after MOABB.

- **Protocol:**
  - Splits are chosen per task: the authors' split if one exists,
    cross-subject (optionally stratified) for clinical tasks, leave-concept-out
    for cognitive decoding, random where data is scarce.
  - Each model runs with three seeds on one train/validation/test partition.
- **Preprocessing for task-specific models:** resampling to 120 Hz, a 0.1–75 Hz
  band-pass, notch filters at 50/60 Hz, robust per-channel scaling, then
  clamping at 20. Foundation models use their own pretraining pipelines.
- **Training:** AdamW (lr 1e-4, weight decay 0.05), cosine schedule with 10%
  warmup, up to 50 epochs, early stopping with patience 10.
- **Metrics:** balanced accuracy for classification, Pearson correlation for
  regression, and a score normalised against the dummy baseline (0 means dummy
  level, 1 means perfect).
- **Riemannian baselines:** these are MOABB's conventions, not straw men. They
  are tangent space plus logistic regression or ridge, with xDAWN covariances
  for ERPs and co-spectra for SSVEP. They are fit on at most 20,000 training
  trials, subsampled stratified by label.
- **Clinical datasets in the Core suite:**

  | Task | Dataset | Subjects | Our equivalent |
  |---|---|---|---|
  | Dementia | Miltiadous2023 | 88 | EEGBench Alzheimers (on004504) |
  | Depression | Mumtaz2018 | 64 | EEGDash nm000114 |
  | Parkinson's | Singh2021 | 129 | Probably the Iowa PD cohort (on004584 and related); not confirmed |
  | Schizophrenia | Albrecht2019 | 77 | None yet |
  | Pathology | TUAB (Lopez 2017) | 2,329 | None yet |
  | Clinical events | TUEV (Harati 2015) | | None yet |

- **No general time series classifiers.** There are no ROCKET-family
  classifiers, DrCIF, QUANT or HC2.
- **Adding models:**
  - Deep models are registered through a YAML file. Braindecode models are
    picked up automatically by class name.
  - Classical models subclass `SklearnBaseline` in
    `neuralbench/baselines.py`. Those are fit once on the training set and
    wrapped as a `torch.nn.Module` for evaluation.
  - An aeon classifier could plug in the same way. It would then see exactly
    the same preprocessed data and splits as the published models.
- **Submitting:**
  - There is no leaderboard submission process. Published results come from
    the authors' own runs.
  - Contributions go through PRs to the monorepo. New features need an issue
    first, and a signed Meta CLA is required.
  - The EEG/EMG Foundation Challenge 2026 is separate, with its own submission
    guide.

## Relevance to aeon-neuro

- MOABB is the only public league table that covers both classic deep learning
  and non-deep-learning methods.
- All six of MOABB's deep learning models are in `BraindecodeClassifier`'s
  `SUPPORTED_MODELS`. MOABB's ShallowConvNet is braindecode's `ShallowFBCSPNet`,
  and its DeepConvNet is `Deep4Net`.
- MOABB's datasets load through braindecode's `MOABBDataset`. Running
  `BraindecodeClassifier` and the Riemann classifiers through MOABB's
  within-session protocol would let us check our numbers against theirs.
- The library has 206 datasets, mostly BCI: 104 motor imagery, 70 P300/ERP,
  18 SSVEP, 8 c-VEP and 3 resting state. It does not cover clinical, sleep or
  emotion data; braindecode loads some of those itself.
- The foundation-model benchmarks matter if we later wrap models such as LaBraM
  or CBraMod.
- NeuralBench is the best place to test general time series classifiers
  against strong baselines. Its Riemannian pipelines and its preprocessed,
  early-stopped deep models answer the objection that EEGBench's EEG-specific
  baselines are weak. All 8 of its task-specific models are also in
  `SUPPORTED_MODELS`, so it can check `BraindecodeClassifier` too.

## Caveats

- The 2018 MOABB date and the competition dates were not checked against
  primary sources.
- OpenEEG-Bench is dated by its EUSIPCO 2026 paper; the exact release date was
  not confirmed, and its leaderboard page did not render when checked.
- The Riemann vs CNN date (about 2022) is an estimate.
- The model lists for AdaBrain-Bench, EEG-Bench and OmniEEG-Bench were not
  confirmed from the papers themselves.
