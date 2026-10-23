import warnings
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError

from rojak.core.data import load_from_folder
from rojak.core.sources import (
    DataLocation,
    FilesLocation,
    TemplateLocation,
    ZarrLocation,
    is_remote,
    open_dataset,
    resolve_paths,
)
from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from pathlib import Path

DATA_LOCATION_ADAPTER: TypeAdapter[DataLocation] = TypeAdapter(DataLocation)


DAYS: tuple[int, ...] = (1, 2, 3)


def make_day(day: int) -> xr.Dataset:
    time = np.asarray([np.datetime64(f"2024-01-{day:02d}T00")])
    return xr.Dataset(
        {"t": (("time", "latitude"), np.full((1, 3), float(day)))},
        coords={"time": time, "latitude": [0.0, 1.0, 2.0]},
    )


@pytest.fixture
def daily_files(tmp_path) -> "Path":
    for day in (1, 2, 3):
        month_dir = tmp_path / "2024" / "01"
        month_dir.mkdir(parents=True, exist_ok=True)
        make_day(day).to_netcdf(month_dir / f"2024-01-{day:02d}.nc", engine="h5netcdf")
    return tmp_path


@pytest.fixture
def memory_files():
    fs = fsspec.filesystem("memory")
    root = "/rojak-test-data"
    for day in (1, 2, 3):
        with fs.open(f"{root}/2024-01-{day:02d}.nc", "wb") as f:
            f.write(make_day(day).to_netcdf(engine="h5netcdf"))
    yield f"memory://{root}"
    fs.rm(root, recursive=True)


def january_window(last_day: int) -> Limits[datetime]:
    return Limits(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, last_day, tzinfo=UTC))


@pytest.mark.parametrize(
    ("urlpath", "expected"),
    [("~/data", False), ("/data", False), ("file:///data", False), ("s3://bucket/data", True), ("memory://x", True)],
)
def test_is_remote(urlpath: str, expected: bool) -> None:
    assert is_remote(urlpath) == expected


def test_location_discriminated_union(tmp_path) -> None:
    assert isinstance(DATA_LOCATION_ADAPTER.validate_python({"type": "files", "root": str(tmp_path)}), FilesLocation)
    assert isinstance(DATA_LOCATION_ADAPTER.validate_python({"type": "zarr", "store": "s3://b/x.zarr"}), ZarrLocation)
    with pytest.raises(ValidationError):
        DATA_LOCATION_ADAPTER.validate_python({"type": "unknown", "root": str(tmp_path)})
    with pytest.raises(ValidationError):  # typo'd fields are rejected rather than silently ignored
        DATA_LOCATION_ADAPTER.validate_python({"type": "files", "root": str(tmp_path), "glob": "*.nc"})


def test_files_location_local_root_must_exist(tmp_path) -> None:
    with pytest.raises(ValidationError):
        FilesLocation(root=str(tmp_path / "missing"))
    # Remote roots are not checked eagerly
    FilesLocation(root="s3://some-bucket/that/may/not/exist")


def test_files_location_resolves_recursive_glob(daily_files) -> None:
    paths = resolve_paths(FilesLocation(root=str(daily_files), glob_pattern="**/*.nc"))
    assert [p.rsplit("/", 1)[-1] for p in paths] == ["2024-01-01.nc", "2024-01-02.nc", "2024-01-03.nc"]
    assert all(not p.startswith("file://") for p in paths)


def test_files_location_no_matches(daily_files) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_paths(FilesLocation(root=str(daily_files), glob_pattern="*.grib"))


@pytest.mark.parametrize(
    ("last_day", "on_missing", "expectation"),
    [
        pytest.param(3, "raise", nullcontext(3), id="all_present"),
        pytest.param(5, "skip", nullcontext(3), id="skip_missing"),
        pytest.param(5, "raise", pytest.raises(FileNotFoundError), id="raise_missing"),
    ],
)
def test_template_location(daily_files, last_day, on_missing, expectation) -> None:
    location = TemplateLocation(
        root=str(daily_files),
        template="{time:%Y}/{time:%m}/{time:%Y-%m-%d}.nc",
        time_window=january_window(last_day),
        on_missing=on_missing,
    )
    with expectation as num_files:
        assert len(resolve_paths(location)) == num_files


def test_template_location_deduplicates(daily_files) -> None:
    location = TemplateLocation(
        root=str(daily_files),
        template="{time:%Y}/{time:%m}/{time:%Y-%m-01}.nc",
        time_window=january_window(3),
        frequency="6h",
    )
    assert len(resolve_paths(location)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"template": "no_time_field.nc", "time_window": january_window(3)}, id="no_time_field"),
        pytest.param({"template": "{time:%Y}.nc", "time_window": january_window(3)[::-1]}, id="decreasing_window"),
    ],
)
def test_template_location_invalid(tmp_path, kwargs) -> None:
    with pytest.raises(ValidationError):
        TemplateLocation(root=str(tmp_path), **kwargs)


def test_open_dataset_local_template(daily_files) -> None:
    location = TemplateLocation(
        root=str(daily_files), template="{time:%Y}/{time:%m}/{time:%Y-%m-%d}.nc", time_window=january_window(2)
    )
    ds = open_dataset(location, chunks={})
    assert ds.sizes["time"] == len(DAYS) - 1
    np.testing.assert_array_equal(ds["t"].isel(latitude=0).to_numpy(), [1.0, 2.0])


def test_open_dataset_requires_chunks_for_files(daily_files) -> None:
    with pytest.raises(ValueError, match="Chunks"):
        open_dataset(FilesLocation(root=str(daily_files), glob_pattern="**/*.nc"), chunks=None)


def test_open_dataset_remote_files(memory_files) -> None:
    ds = open_dataset(FilesLocation(root=memory_files), chunks={})
    assert ds.sizes["time"] == len(DAYS)
    np.testing.assert_array_equal(ds["t"].isel(latitude=0).to_numpy(), [1.0, 2.0, 3.0])


def test_open_dataset_remote_files_requires_h5netcdf(memory_files) -> None:
    with pytest.raises(ValueError, match="h5netcdf"):
        open_dataset(FilesLocation(root=memory_files), chunks={}, engine="netcdf4")


def test_open_dataset_zarr(tmp_path) -> None:
    store = tmp_path / "data.zarr"
    xr.concat([make_day(day) for day in (1, 2)], dim="time").to_zarr(store, consolidated=False)
    ds = open_dataset(ZarrLocation(store=str(store), consolidated=False), chunks=None)
    assert ds.sizes["time"] == len(DAYS) - 1


def test_load_from_folder_is_backwards_compatible(daily_files) -> None:
    ds = load_from_folder(daily_files / "2024" / "01", chunks={}, engine="h5netcdf")
    assert ds.sizes["time"] == len(DAYS)
    with pytest.raises(ValueError, match="Chunks"):
        load_from_folder(daily_files / "2024" / "01", chunks=None)


@pytest.fixture
def misaligned_files(tmp_path) -> "Path":
    make_day(1).to_netcdf(tmp_path / "2024-01-01.nc", engine="h5netcdf")
    make_day(2).assign_coords(latitude=[0.0, 1.0, 3.0]).to_netcdf(tmp_path / "2024-01-02.nc", engine="h5netcdf")
    return tmp_path


def test_open_dataset_files_does_not_warn_about_xarray_defaults(misaligned_files) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        ds = open_dataset(FilesLocation(root=str(misaligned_files)), chunks={})
    # Default join="outer" keeps the union of the misaligned coordinates
    assert ds.sizes["latitude"] == 4  # noqa: PLR2004
