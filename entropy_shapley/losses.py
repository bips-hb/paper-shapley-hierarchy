"""Distribution-loss import paths used by stored DeepAR checkpoints.

The checkpoints in models/ were saved with these class paths; this module keeps
them loadable. The losses themselves live in experiments/electricity/deepar_losses.py.
"""

from experiments.electricity.deepar_losses import (
    LogNormalDistributionLoss,
    StudentTDistributionLoss,
)

__all__ = ["StudentTDistributionLoss", "LogNormalDistributionLoss"]
