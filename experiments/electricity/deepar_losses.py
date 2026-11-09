# =============================================================================
#                           DeepAR Distribution Losses
#
# Custom one-step likelihoods for the Electricity DeepAR models.
# PyTorch Forecasting supplies the Normal likelihood; this file defines the
# Student-t and LogNormal variants used in the estimator validation (App. D.3.3).
# The stored checkpoints refer to these classes by name; do not rename them.
#
# Parameter transformations must agree with the corresponding entropy
# imputers. Affine target scaling is supported; additional nonlinear
# target-normalizer transformations are rejected.
# =============================================================================

import torch
import torch.nn.functional as F
from torch import distributions
from torch.distributions import transforms as T

from pytorch_forecasting.metrics import DistributionLoss


# =============================================================================
#                              Student-t likelihood
# =============================================================================

class StudentTDistributionLoss(DistributionLoss):
    """Student-t likelihood with affine target scaling.

    The network emits (df_raw, loc, scale_raw). Parameters are transformed
    as df = softplus(df_raw) + 2 and scale = softplus(scale_raw), ensuring
    finite variance and positive scale. The target affine transform is
    applied to the resulting Student-t variable.
    """

    distribution_class = distributions.StudentT
    distribution_arguments = ["df", "loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.StudentT:
        """Build the predictive distribution from scaled parameters (..., 5).

        The final axis stores [target_loc, target_scale, df, loc, scale].
        """
        distr = distributions.StudentT(df=x[..., 2], loc=x[..., 3], scale=x[..., 4])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler])
        else:
            raise NotImplementedError("StudentTDistributionLoss does not yet support "
                                      "target normalizers with transformations.")

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder,
    ) -> torch.Tensor:
        """Transform raw outputs (B, T, 3) and attach target scaling (B, 2).

        Return (B, T, 5) with layout [target_loc, target_scale, df, loc, scale].
        """
        self._transformation = encoder.transformation
        df    = F.softplus(parameters[..., 0]) + 2  # enforce df > 2 for finite variance
        loc   = parameters[..., 1]
        scale = F.softplus(parameters[..., 2])
        return torch.concat([
            target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
            df.unsqueeze(-1),
            loc.unsqueeze(-1),
            scale.unsqueeze(-1),
        ], dim=-1)


# =============================================================================
#                              LogNormal likelihood
# =============================================================================

class LogNormalDistributionLoss(DistributionLoss):
    """LogNormal likelihood obtained by exponentiating an affine-scaled Normal.

    The network emits (loc, scale_raw), with
    scale = sigmoid(scale_raw) + 1e-3. For Z ~ Normal(loc, scale), the
    predictive variable is exp(target_loc + target_scale * Z).
    This scale transformation also defines the LogNormal entropy imputer.
    """

    distribution_class = distributions.Normal
    distribution_arguments = ["loc", "scale"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.TransformedDistribution:
        """Build the predictive distribution from scaled parameters (..., 4).

        The final axis stores [target_loc, target_scale, loc, scale]. Apply
        the affine transform in log space, followed by exponentiation.
        """
        distr = distributions.Normal(loc=x[..., 2], scale=x[..., 3])
        scaler = distributions.AffineTransform(loc=x[..., 0], scale=x[..., 1])
        if self._transformation is None:
            return distributions.TransformedDistribution(distr, [scaler, T.ExpTransform()])
        else:
            raise NotImplementedError("LogNormalDistributionLoss does not yet support "
                                      "target normalizers with transformations.")

    def rescale_parameters(
        self,
        parameters: torch.Tensor,
        target_scale: torch.Tensor,
        encoder,
    ) -> torch.Tensor:
        """Transform raw outputs (B, T, 2) and attach target scaling (B, 2).

        Return (B, T, 4) with layout [target_loc, target_scale, loc, scale].
        """
        self._transformation = encoder.transformation
        loc   = parameters[..., 0]
        # This scale transformation also defines the LogNormal entropy imputer.
        scale = torch.sigmoid(parameters[..., 1]) * 1.0 + 1e-3
        return torch.concat([
            target_scale.unsqueeze(1).expand(-1, loc.size(1), -1),
            loc.unsqueeze(-1),
            scale.unsqueeze(-1),
        ], dim=-1)

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """Compute elementwise negative log-likelihood, clamping targets to 1e-6."""
        distribution = self.map_x_to_distribution(y_pred)
        return -distribution.log_prob(y_actual.clamp(min=1e-6))
