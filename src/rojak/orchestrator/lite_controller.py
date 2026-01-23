from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rojak.orchestrator.lite_configuration import DistributionParametersContext


def compute_distribution_parameters(context: "DistributionParametersContext") -> None:
    print(context)
