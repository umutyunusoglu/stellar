from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt


class Distribution(Protocol):
    """Any continuous distribution with a cdf and its inverse.

    Every scipy.stats frozen distribution fits, including a custom
    rv_continuous that defines only _pdf; scipy derives the rest.
    """

    def cdf(self, x: npt.ArrayLike) -> npt.ArrayLike: ...

    def ppf(self, q: npt.ArrayLike) -> npt.ArrayLike: ...


@dataclass
class CostFunction:
    """
    How damage multipliers are distributed.

    A multiplier is described by its severity: its quantile in the damage
    distribution conditioned on being at least 1, so severity 0 is no extra
    cost and severity 1 the worst. The map is monotone, so a worse condition
    always costs more. It knows nothing about damage states: a discrete
    model assigns each state a band of severities, a continuous model can
    produce severities directly.

    The two directions are computed from the distribution independently:
    multiplier uses its inverse cdf, severity its cdf.

    attributes:
        distribution: The multiplier distribution. Only its part above 1 is
            used, so it must put some mass there.
    """

    distribution: Distribution

    _floor: float = field(init=False)

    def __post_init__(self) -> None:
        self._floor = float(np.asarray(self.distribution.cdf(1.0)))
        if self._floor >= 1.0:
            raise ValueError("distribution has no mass above multiplier 1")

    def multiplier(self, severity: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Map severities to multipliers.

        Params:
            severity: One or more severities in [0, 1].

        Returns:
            The multipliers, at least 1, with the shape of the input.
        """
        q = self._floor + np.asarray(severity, dtype=np.float64) * (1.0 - self._floor)
        return np.maximum(np.asarray(self.distribution.ppf(q), dtype=np.float64), 1.0)

    def severity(self, multiplier: float) -> float:
        """Map a multiplier back to its severity.

        Params:
            multiplier: An observed multiplier.

        Returns:
            Its severity, clamped to [0, 1]; anything up to 1 gives 0.
        """
        if multiplier <= 1.0:
            return 0.0
        q = float(np.asarray(self.distribution.cdf(multiplier)))
        return min(max((q - self._floor) / (1.0 - self._floor), 0.0), 1.0)
