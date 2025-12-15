from typing import TYPE_CHECKING, cast

import dask.dataframe as dd
import numpy as np

from rojak.cli.main import app
from rojak.datalib.madis.amdar import AcarsAmdarRepository, AcarsAmdarTurbulenceData
from rojak.orchestrator.configuration import SpatialDomain
from tests.test_cli import runner

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_climatological_edr(retrieve_single_day_madis_data):
    spatial_domain = SpatialDomain(
        minimum_latitude=-90,
        maximum_latitude=90,
        minimum_longitude=-180,
        maximum_longitude=180,
        grid_size=0.25,
    )
    acars_data: AcarsAmdarTurbulenceData = cast(
        "AcarsAmdarTurbulenceData",
        AcarsAmdarRepository(str(retrieve_single_day_madis_data)).to_amdar_turbulence_data(
            spatial_domain,
            0.25,
            [175, 200, 225, 250, 300, 350],
        ),
    )
    edr_distribution = acars_data.edr_distribution()
    max_edr: NDArray = acars_data.data_frame["maxEDR"].to_dask_array().compute()
    max_edr = max_edr[max_edr > 0]
    ln_max_edr = np.log(max_edr)
    # Fails in CI where 7th decimal is different actual is 3, desired is 2
    np.testing.assert_almost_equal(edr_distribution.mean, float(np.nanmean(ln_max_edr)))
    np.testing.assert_almost_equal(edr_distribution.variance, float(np.nanvar(ln_max_edr)))


def test_repartition_parquet_files(retrieve_single_day_madis_data, tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("output")
    num_new_partitions: int = 3
    repartition_invoke = runner.invoke(
        app,
        [
            "data",
            "utils",
            "repartition",
            "-r",
            "-d",
            f"{retrieve_single_day_madis_data!s}/",
            "-o",
            str(output_dir),
            "-p",
            "**/*.parquet",
            "-n",
            f"{num_new_partitions}",
        ],
    )
    assert repartition_invoke.exit_code == 0
    root_output_folder = output_dir / "2024"
    assert root_output_folder.exists()
    # Check that only 3 part files are created
    for i in range(3):
        partition_file = root_output_folder / f"part.{i}.parquet"
        assert partition_file.exists()
        assert partition_file.is_file()
    assert dd.read_parquet(str(root_output_folder / "**/*.parquet")).npartitions == num_new_partitions
