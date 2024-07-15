<p align="center">
    <a href="https://aeon-toolkit.org"><img src="https://raw.githubusercontent.com/aeon-toolkit/aeon/main/docs/images/logo/aeon-logo-blue-compact.png" width="50%" alt="aeon logo" /></a>
</p>

# ⌛ Welcome to aeon-neuro

`aeon-neuro` is package that unifies techniques for the classification of EEG time
series. Our goal is to present a simple and unified interface to a variety of EEG
classification problems that combine techniques from a range of domains that learn
from EEG signals.

We aim to develop this package following the principles of open science, and
reproducible research, as described in the [Turing Way](https://github.com/the-turing-way) and this package is based on [this template](https://github.com/the-turing-way/reproducible-project-template).

`aeon-neuro` is a companion package to the `aeon` toolkit. The main project webpage
and documentation is available at https://aeon-toolkit.org and the source code at
https://github.com/aeon-toolkit/aeon.

The initial `aeon-neuro` release is `v0.0.1`.

Our webpage and documentation is available at https://aeon-neuro.readthedocs.io.

| Overview      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**     | [![github-actions-release](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/release.yml?logo=github&label=build%20%28release%29)](https://github.com/aeon-toolkit/aeon-neuro/actions/workflows/release.yml) [![github-actions-main](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/pr_pytest.yml?logo=github&branch=main&label=build%20%28main%29)](https://github.com/aeon-toolkit/aeon-neuro/actions/workflows/pr_pytest.yml) [![github-actions-nightly](https://img.shields.io/github/actions/workflow/status/aeon-toolkit/aeon-neuro/periodic_tests.yml?logo=github&label=build%20%28nightly%29)](https://github.com/aeon-toolkit/aeon-neuro/actions/workflows/periodic_tests.yml) [![docs-main](https://img.shields.io/readthedocs/aeon-neuro/stable?logo=readthedocs&label=docs%20%28stable%29)](https://aeon-neuro.readthedocs.io/en/stable/?badge=stable) [![docs-main](https://img.shields.io/readthedocs/aeon-neuro/latest?logo=readthedocs&label=docs%20%28latest%29)](https://aeon-neuro.readthedocs.io/en/latest/?badge=latest) [![!codecov](https://img.shields.io/codecov/c/github/aeon-toolkit/aeon-neuro?label=codecov&logo=codecov)](https://codecov.io/gh/aeon-toolkit/aeon-neuro) |
| **Code**      | [![!pypi](https://img.shields.io/pypi/v/aeon-neuro?logo=pypi&color=blue)](https://pypi.org/project/aeon-neuro/) [![!python-versions](https://img.shields.io/pypi/pyversions/aeon-neuro?logo=python)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![license](https://img.shields.io/badge/license-BSD%203--Clause-green?logo=style)](https://github.com/aeon-toolkit/aeon/blob/main/LICENSE)                                                                                                                                                                                                                                                                                                                                                                                                                             |
| **Community** | [![!slack](https://img.shields.io/static/v1?logo=slack&label=Slack%20%28aeon-neuro%29&message=chat&color=lightgreen)](https://join.slack.com/t/aeon-neuro/shared_invite/zt-2k4qs8mjb-ZZs~6P0MdF8kGf9cUQzKSg) [![!slack-aeon](https://img.shields.io/static/v1?logo=slack&label=Slack%20%28aeon%29&message=chat&color=lightgreen)](https://join.slack.com/t/aeon-toolkit/shared_invite/zt-22vwvut29-HDpCu~7VBUozyfL_8j3dLA) [![!linkedin](https://img.shields.io/static/v1?logo=linkedin&label=LinkedIn&message=news&color=lightblue)](https://www.linkedin.com/company/aeon-toolkit/) [![!twitter](https://img.shields.io/static/v1?logo=twitter&label=Twitter&message=news&color=lightblue)](https://twitter.com/aeon_toolkit)                                                                                                                                                                           |

## ⚙️ Installation

`aeon-neuro` requires a Python version of 3.9 or greater. Our full installation guide is available in our [documentation](https://aeon-neuro.readthedocs.io/en/latest/installation.html).

The easiest way to install `aeon` is via pip:

```bash
pip install aeon-neuro
```

Some estimators require additional packages to be installed. If you want to install
the full package with all optional dependencies, you can use:

```bash
pip install aeon-neuro[all_extras]
```

Instructions for installation from the [GitHub source](https://github.com/aeon-toolkit/aeon-neuro) can be found [here](https://aeon-neuro.readthedocs.io/en/latest/installation.html#install-the-latest-development-version-using-pip).


## 💬 Where to ask questions

| Type                                | Platforms                        |
|-------------------------------------|----------------------------------|
| 🐛 **Bug Reports**                  | [GitHub Issue Tracker]           |
| ✨ **Feature Requests & Ideas**      | [GitHub Issue Tracker] & [Slack] |
| 💻 **Usage Questions**              | [GitHub Discussions] & [Slack]   |
| 💬 **General Discussion**           | [GitHub Discussions] & [Slack]   |
| 🏭 **Contribution & Development**   | [Slack]                          |

[GitHub Issue Tracker]: https://github.com/aeon-toolkit/aeon-neuro/issues
[GitHub Discussions]: https://github.com/aeon-toolkit/aeon-neuro/discussions
[Slack]: https://join.slack.com/t/aeon-neuro/shared_invite/zt-2k4qs8mjb-ZZs~6P0MdF8kGf9cUQzKSg


## 💡 Acknowledgements
This work is supported by the UK Engineering and Physical Sciences Research Council
(EPSRC) EP/W030756/2
