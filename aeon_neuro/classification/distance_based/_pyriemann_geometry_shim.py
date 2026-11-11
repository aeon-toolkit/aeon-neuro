"""Let unmodified TRACE (aeon-neuro PR #296) run on pyriemann 0.10 or 0.11.

TRACE imports ``pyriemann.geometry``, which first appears in pyriemann 0.12.
In earlier releases the same functions live in ``pyriemann.utils``. Import
this module before importing TRACE; it does nothing on pyriemann >= 0.12.

Checked on EyesOpenShut, FingerMovements and Blink: TRACE's predict_proba
output is bit-for-bit identical under pyriemann 0.12 and under 0.10 with this
shim.
"""

import sys
import types

import pyriemann

try:
    import pyriemann.geometry  # noqa: F401  (pyriemann >= 0.12)
except ImportError:
    import pyriemann.utils.covariance as _covariance
    import pyriemann.utils.distance as _distance
    import pyriemann.utils.mean as _mean

    _geometry = types.ModuleType("pyriemann.geometry")
    _geometry.covariance = _covariance
    _geometry.distance = _distance
    _geometry.mean = _mean
    pyriemann.geometry = _geometry
    sys.modules["pyriemann.geometry"] = _geometry
    sys.modules["pyriemann.geometry.covariance"] = _covariance
    sys.modules["pyriemann.geometry.distance"] = _distance
    sys.modules["pyriemann.geometry.mean"] = _mean
