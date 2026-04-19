<p align="center">
    <a href="https://aeon-toolkit.org">
        <img
            src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logo/aeon-logo-blue-compact.png"
            width="50%"
            alt="aeon logo"
        />
    </a>
</p>
<h1 align="center">aeon-neuro</h1>
<p align="center">
    <strong>Reproducible EEG classification, built on modern time series machine learning.</strong>
</p>
<p align="center">
    <a href="https://aeon-neuro.readthedocs.io">Documentation</a> ·
    <a href="https://github.com/aeon-toolkit/aeon-neuro">Source code</a> ·
    <a href="https://pypi.org/project/aeon-neuro/">PyPI</a> ·
    <a href="https://github.com/aeon-toolkit/aeon-neuro/discussions">Discussions</a>
</p>
EEG classification research is fragmented. Methods are spread across signal processing,
deep learning, BCI, and general machine learning toolkits, often with different data
formats, bespoke pipelines, and evaluation protocols.
`aeon-neuro` brings these strands together in a single open-source package for EEG time
series classification. It is designed to make experiments easier to run, easier to compare,
and easier to reproduce. The project provides a unified interface for EEG classification
problems, benchmark datasets, and baseline methods, while building on the design
principles of `aeon`, the Python toolkit for time series machine
learning.
Whether you are testing a new EEG method, benchmarking against strong baselines, or
trying to reproduce published results, `aeon-neuro` aims to give you a cleaner starting
point.
Overview	
CI/CD	![github-actions-release](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/release.yml?logo=github&label=build%20%28release%29) ![github-actions-main](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/pr_pytest.yml?logo=github&branch=main&label=build%20%28main%29) ![github-actions-nightly](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/periodic_tests.yml?logo=github&label=build%20%28nightly%29) ![docs-stable](https://img.shields.io/readthedocs/aeon-neuro/stable?logo=readthedocs&label=docs%20%28stable%29) ![docs-latest](https://img.shields.io/readthedocs/aeon-neuro/latest?logo=readthedocs&label=docs%20%28latest%29) ![codecov](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon-neuro?label=codecov&logo=codecov)
Code	![pypi](https://img.shields.io/pypi/v/aeon-neuro?logo=pypi&color=blue) ![conda](https://img.shields.io/conda/vn/conda-forge/aeon-neuro?logo=anaconda&color=blue) ![python-versions](https://img.shields.io/pypi/pyversions/aeon-neuro?logo=python) ![black](https://img.shields.io/badge/code%20style-black-000000.svg) ![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)
Community	![slack-neuro](https://img.shields.io/static/v1?logo=slack&label=Slack%20%28aeon-neuro%29&message=chat&color=lightgreen) ![slack-aeon](https://img.shields.io/static/v1?logo=slack&label=Slack%20%28aeon%29&message=chat&color=lightgreen) ![linkedin](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue) ![twitter](https://img.shields.io/static/v1?logo=twitter&label=Twitter&message=news&color=lightblue)
Why aeon-neuro?
A unified interface for EEG classification  
Work with EEG classification problems through a consistent API instead of stitching
together multiple incompatible workflows.
Reproducible benchmarks  
Run experiments on curated EEG classification problems with fixed train/test splits and
shared evaluation protocols.
Strong baselines out of the box  
Compare against established methods from time series machine learning, deep learning,
and related EEG workflows.
Built with open science in mind  
The package is developed to support transparent, reproducible, and extensible research.
Part of the aeon ecosystem  
`aeon-neuro` is a companion package to `aeon`, extending
its time series machine learning foundations to EEG applications.
What this project is for
`aeon-neuro` is intended for researchers and practitioners who want to:
benchmark new EEG classifiers against strong baselines,
reproduce published EEG classification experiments,
load and evaluate EEG benchmark problems consistently,
test general-purpose time series classifiers on EEG data,
build more reliable experimental pipelines for EEG research.
The project is especially motivated by a simple problem in the field: EEG classification is
widely studied, but comparisons are often hard because data preparation, feature
engineering, model choices, and evaluation protocols vary so much from paper to paper.
Benchmark-first design
A central goal of `aeon-neuro` is to support fairer and easier evaluation.
The project is tied to a benchmark archive of EEG classification problems spanning medical,
brain-computer interface, and psychology applications. This makes it possible to compare
methods across a broader range of tasks, rather than drawing conclusions from a small
hand-picked subset of datasets.
This also makes `aeon-neuro` useful beyond software alone. It is a research scaffold for
building stronger baselines, reproducing studies, and understanding where different classes
of method work well or fail.
Installation
`aeon-neuro` requires Python 3.9 or later. Full installation instructions are available in the
documentation.
Install the core package with pip:
```bash
pip install aeon-neuro
```
To install with all optional dependencies:
```bash
pip install aeon-neuro[all_extras]
```
To install the latest development version from GitHub, see the
installation guide.
Documentation
Project documentation: https://aeon-neuro.readthedocs.io
aeon documentation: https://aeon-toolkit.org
Source code: https://github.com/aeon-toolkit/aeon-neuro
Where to ask questions
Type	Platforms
🐛 Bug reports	GitHub Issue Tracker
✨ Feature requests and ideas	GitHub Issue Tracker and Slack
💻 Usage questions	GitHub Discussions and Slack
💬 General discussion	GitHub Discussions and Slack
🏭 Contribution and development	Slack
Contributing
We welcome contributions across code, datasets, benchmarking, documentation, and testing.
If you want to contribute, start with:
the issue tracker,
the discussions page,
or the community Slack.
Acknowledgements
This work is supported by the UK Engineering and Physical Sciences Research Council
(EPSRC) under grant EP/W030756/2.