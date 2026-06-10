# Detach-ROCKET attribution

The Sequential Feature Detachment algorithm in `_detach_rocket.py` is adapted
from:

- https://github.com/gon-uri/detach_rocket
- Upstream commit: `aa046a3`
- Authors: Gonzalo Uribarri, Federico Barone, and contributors

The upstream README identifies the project license as BSD-3-Clause. The adapted
implementation replaces sktime ROCKET transformers with aeon MiniRocket and
integrates the algorithm as an aeon channel selector.
