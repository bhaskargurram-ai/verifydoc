"""Post-hoc calibrators (PROJECT.md §5.G), fit ONLY on the calibration split."""

from verifydoc.calibration.base import Calibrator
from verifydoc.calibration.characterization import (
    CharacterizationReport,
    characterize,
    predict_coverage_gain,
)
from verifydoc.calibration.conformal import ConformalAbstention
from verifydoc.calibration.grouped_conformal import (
    GROUP_TAXONOMIES,
    GroupConformalAbstention,
    GroupPartitionSelector,
    combine_groups,
    field_type_group,
    grounded_group,
    support_bin_group,
    value_length_group,
)
from verifydoc.calibration.histogram import HistogramBinning
from verifydoc.calibration.isotonic import IsotonicCalibrator
from verifydoc.calibration.platt import PlattScaling
from verifydoc.calibration.splits import assert_disjoint, split_calibration
from verifydoc.calibration.temperature import TemperatureScaling

__all__ = [
    "Calibrator",
    "CharacterizationReport",
    "ConformalAbstention",
    "GROUP_TAXONOMIES",
    "GroupConformalAbstention",
    "GroupPartitionSelector",
    "HistogramBinning",
    "IsotonicCalibrator",
    "PlattScaling",
    "TemperatureScaling",
    "assert_disjoint",
    "characterize",
    "combine_groups",
    "field_type_group",
    "grounded_group",
    "predict_coverage_gain",
    "split_calibration",
    "support_bin_group",
    "value_length_group",
]
