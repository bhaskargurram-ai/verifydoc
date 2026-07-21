"""Post-hoc calibrators (PROJECT.md §5.G), fit ONLY on the calibration split."""

from verifydoc.calibration.base import Calibrator
from verifydoc.calibration.conformal import ConformalAbstention
from verifydoc.calibration.histogram import HistogramBinning
from verifydoc.calibration.isotonic import IsotonicCalibrator
from verifydoc.calibration.platt import PlattScaling
from verifydoc.calibration.splits import assert_disjoint, split_calibration
from verifydoc.calibration.temperature import TemperatureScaling

__all__ = [
    "Calibrator",
    "ConformalAbstention",
    "HistogramBinning",
    "IsotonicCalibrator",
    "PlattScaling",
    "TemperatureScaling",
    "assert_disjoint",
    "split_calibration",
]
