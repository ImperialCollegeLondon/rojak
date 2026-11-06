import warnings
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import TypeAdapter, ValidationError

from rojak.core.data import load_from_folder
from rojak.core.sources import (
    DataLocation,
    FilesLocation,
    MeteorologicalSeason,
    NetcdfEngine,
    ParquetLocation,
    TemplateLocation,
    ZarrLocation,
    _construct_seasonal_filter,
    is_remote,
    open_dataframe,
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
        open_dataset(FilesLocation(root=memory_files), chunks={}, engine=NetcdfEngine.NETCDF4)


def test_open_dataset_zarr(tmp_path) -> None:
    store = tmp_path / "data.zarr"
    xr.concat([make_day(day) for day in (1, 2)], dim="time").to_zarr(store, consolidated=False)
    ds = open_dataset(ZarrLocation(store=str(store), consolidated=False), chunks=None)
    assert ds.sizes["time"] == len(DAYS) - 1


def test_load_from_folder_is_backwards_compatible(daily_files) -> None:
    ds = load_from_folder(daily_files / "2024" / "01", chunks={}, engine=NetcdfEngine.H5NETCDF)
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


def test_open_dataset_open_kwargs_override_defaults(misaligned_files) -> None:
    with pytest.raises(ValueError, match="join='exact'"):
        open_dataset(FilesLocation(root=str(misaligned_files)), chunks={}, join="exact")


def test_open_dataset_open_kwargs_passed_to_open_zarr(tmp_path) -> None:
    store = tmp_path / "data.zarr"
    make_day(1).assign(u=lambda ds: ds["t"] * 2).to_zarr(store, consolidated=False)
    ds = open_dataset(ZarrLocation(store=str(store), consolidated=False), chunks=None, drop_variables=["u"])
    assert set(ds.data_vars) == {"t"}


def test_load_from_folder_passes_open_kwargs(misaligned_files) -> None:
    with pytest.raises(ValueError, match="join='exact'"):
        load_from_folder(misaligned_files, chunks={}, engine=NetcdfEngine.H5NETCDF, join="exact")


def utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


def naive(moment: datetime) -> datetime:
    """Drop the timezone, as a time window from configuration is not guaranteed to carry one"""
    return moment.replace(tzinfo=None)


def window(lower: datetime, upper: datetime, *, is_aware: bool = True) -> Limits[datetime]:
    return Limits(lower, upper) if is_aware else Limits(naive(lower), naive(upper))


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def periods(time_filter) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Inclusive start and exclusive end of each period of a disjunctive normal form filter"""
    return [(group[0][2], group[1][2]) for group in time_filter]


def select(time_filter, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Apply a pyarrow filter the way pyarrow would: an OR over groups of ANDed predicates"""
    groups = time_filter if isinstance(time_filter[0], list) else [time_filter]
    matched = np.zeros(len(timestamps), dtype=bool)
    for group in groups:
        in_group = np.ones(len(timestamps), dtype=bool)
        for _, operator, value in group:
            in_group &= (timestamps >= value) if operator == ">=" else (timestamps < value)
        matched |= in_group
    return matched


def brute_force(window_to_select: Limits[datetime], season: MeteorologicalSeason | None, stamps) -> np.ndarray:
    """Independent check: a timestamp is kept if it is within the inclusive window and in the season"""
    kept = (stamps >= pd.Timestamp(window_to_select.lower)) & (
        stamps < pd.Timestamp(window_to_select.upper + timedelta(days=1))
    )
    if season is not None:
        kept &= ((stamps.month % 12 + 3) // 3) == season.as_int
    return kept


# Three-hourly so that a boundary shifted by a time of day is visible
SEASONAL_STAMPS: pd.DatetimeIndex = pd.date_range("2019-10-01", "2022-06-01", freq="3h", tz="UTC")
NAIVE_SEASONAL_STAMPS: pd.DatetimeIndex = SEASONAL_STAMPS.tz_localize(None)


def stamps_for(time_window: Limits[datetime]) -> pd.DatetimeIndex:
    """Timestamps to filter, matching the awareness of the window: the two cannot be compared otherwise"""
    return SEASONAL_STAMPS if time_window.lower.tzinfo is not None else NAIVE_SEASONAL_STAMPS


class TestConstructSeasonalFilter:
    def test_groups_adjacent_months_into_one_period(self) -> None:
        filters = _construct_seasonal_filter(
            window(utc(2020, 1, 1), utc(2020, 12, 31)), time_column="t", season=MeteorologicalSeason.DJF
        )
        # January and February are adjacent, December starts a separate period
        assert periods(filters) == [
            (ts("2020-01-01"), ts("2020-03-01")),
            (ts("2020-12-01"), ts("2021-01-01")),
        ]

    def test_merges_season_spanning_year_boundary(self) -> None:
        filters = _construct_seasonal_filter(
            window(utc(2020, 12, 1), utc(2021, 2, 28)), time_column="t", season=MeteorologicalSeason.DJF
        )
        assert periods(filters) == [(ts("2020-12-01"), ts("2021-03-01"))]

    def test_clips_period_to_upper_limit(self) -> None:
        # Without clipping the December period would run to 2021-02-01, past the window
        filters = _construct_seasonal_filter(
            window(utc(2020, 1, 1), utc(2021, 1, 20)), time_column="t", season=MeteorologicalSeason.DJF
        )
        assert periods(filters)[-1] == (ts("2020-12-01"), ts("2021-01-21"))

    def test_keeps_partially_covered_first_month(self) -> None:
        # The window starts mid-January, which is still a DJF month
        filters = _construct_seasonal_filter(
            window(utc(2020, 1, 15), utc(2020, 3, 31)), time_column="t", season=MeteorologicalSeason.DJF
        )
        assert periods(filters) == [(ts("2020-01-15"), ts("2020-03-01"))]

    def test_time_of_day_does_not_shift_period_boundaries(self) -> None:
        # A lower limit with a time of day must not carry that time into the month boundaries, which would
        # leak the first hours of March into DJF and lose them from MAM
        lower = utc(2020, 1, 1, 6)
        djf = _construct_seasonal_filter(
            window(lower, utc(2020, 7, 31)), time_column="t", season=MeteorologicalSeason.DJF
        )
        mam = _construct_seasonal_filter(
            window(lower, utc(2020, 7, 31)), time_column="t", season=MeteorologicalSeason.MAM
        )
        assert periods(djf) == [(pd.Timestamp(lower), ts("2020-03-01"))]
        assert periods(mam) == [(ts("2020-03-01"), ts("2020-06-01"))]

    @pytest.mark.parametrize("is_aware", [True, False], ids=["aware", "naive"])
    def test_period_bounds_match_window_awareness(self, is_aware: bool) -> None:
        # A time window parsed from configuration need not carry a timezone, and must behave the same way
        filters = _construct_seasonal_filter(
            window(utc(2020, 1, 1, 6), utc(2020, 3, 31), is_aware=is_aware),
            time_column="t",
            season=MeteorologicalSeason.DJF,
        )
        expected_start = ts("2020-01-01 06:00") if is_aware else pd.Timestamp("2020-01-01 06:00")
        expected_end = ts("2020-03-01") if is_aware else pd.Timestamp("2020-03-01")
        assert periods(filters) == [(expected_start, expected_end)]

    def test_empty_when_window_has_no_month_of_season(self) -> None:
        filters = _construct_seasonal_filter(
            window(utc(2020, 6, 1), utc(2020, 7, 31)), time_column="t", season=MeteorologicalSeason.DJF
        )
        assert filters == []

    def test_uses_time_column_in_every_predicate(self) -> None:
        filters = _construct_seasonal_filter(
            window(utc(2020, 1, 1), utc(2020, 12, 31)),
            time_column="valid_time",
            season=MeteorologicalSeason.DJF,
        )
        assert {column for group in filters for column, _, _ in group} == {"valid_time"}
        assert {operator for group in filters for _, operator, _ in group} == {">=", "<"}

    @pytest.mark.parametrize("is_aware", [True, False], ids=["aware", "naive"])
    @pytest.mark.parametrize("season", list(MeteorologicalSeason))
    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            pytest.param(utc(2020, 1, 1), utc(2021, 12, 31), id="whole_months"),
            pytest.param(utc(2020, 1, 15), utc(2021, 12, 31), id="lower_mid_month"),
            pytest.param(utc(2020, 1, 1), utc(2021, 1, 20), id="upper_mid_month"),
            pytest.param(utc(2020, 1, 1, 6), utc(2021, 3, 1, 9, 45), id="time_of_day"),
            pytest.param(utc(2020, 2, 29), utc(2020, 11, 30), id="leap_day_to_month_end"),
            pytest.param(utc(2019, 11, 1), utc(2022, 3, 1), id="spans_three_winters"),
        ],
    )
    def test_matches_brute_force_selection(self, lower: datetime, upper: datetime, season, is_aware: bool) -> None:
        selected_window = window(lower, upper, is_aware=is_aware)
        stamps = stamps_for(selected_window)
        filters = _construct_seasonal_filter(selected_window, time_column="t", season=season)
        expected = brute_force(selected_window, season, stamps)
        assert np.array_equal(select(filters, stamps), expected)


class TestParquetLocationValidation:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            pytest.param(
                {"time_window": window(utc(2021, 1, 1), utc(2020, 1, 1)), "time_column": "t"},
                "increasing from lower to upper",
                id="decreasing_window",
            ),
            pytest.param(
                {"time_window": window(utc(2020, 1, 1), utc(2020, 12, 31))},
                "Time column must be specified",
                id="window_without_time_column",
            ),
            pytest.param(
                {"season": MeteorologicalSeason.DJF, "time_column": "t"},
                "Time window must be specified",
                id="season_without_window",
            ),
            pytest.param(
                {
                    "time_window": window(utc(2020, 6, 1), utc(2020, 7, 31)),
                    "season": MeteorologicalSeason.DJF,
                    "time_column": "t",
                },
                "contains no djf months",
                id="season_absent_from_window",
            ),
            pytest.param(
                {
                    "time_window": window(utc(2020, 6, 1), utc(2020, 7, 31), is_aware=False),
                    "season": MeteorologicalSeason.DJF,
                    "time_column": "t",
                },
                "contains no djf months",
                id="season_absent_from_naive_window",
            ),
        ],
    )
    def test_invalid(self, tmp_path, kwargs: dict[str, Any], message: str) -> None:
        with pytest.raises(ValidationError, match=message):
            ParquetLocation(root=str(tmp_path), file_name="data.parquet", **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="no_filtering"),
            pytest.param(
                {"time_window": window(utc(2020, 1, 1), utc(2020, 12, 31)), "time_column": "t"},
                id="window_only",
            ),
            pytest.param(
                {
                    "time_window": window(utc(2020, 1, 1), utc(2020, 12, 31)),
                    "season": MeteorologicalSeason.DJF,
                    "time_column": "t",
                },
                id="window_and_season",
            ),
            pytest.param(
                {
                    "time_window": window(utc(2020, 1, 1), utc(2020, 12, 31), is_aware=False),
                    "season": MeteorologicalSeason.DJF,
                    "time_column": "t",
                },
                id="naive_window_and_season",
            ),
        ],
    )
    def test_valid(self, tmp_path, kwargs: dict[str, Any]) -> None:
        assert ParquetLocation(root=str(tmp_path), file_name="data.parquet", **kwargs).file_name == "data.parquet"


class TestParquetLocationTimeFilter:
    def test_none_without_time_window(self, tmp_path) -> None:
        assert ParquetLocation(root=str(tmp_path), file_name="data.parquet").time_filter() is None

    def test_window_without_season_keeps_whole_final_day(self, tmp_path) -> None:
        location = ParquetLocation(
            root=str(tmp_path),
            file_name="data.parquet",
            time_window=window(utc(2020, 1, 1), utc(2020, 3, 31)),
            time_column="t",
        )
        # The window is inclusive, so the exclusive upper predicate is the day after
        assert location.time_filter() == [
            ("t", "<", utc(2020, 4, 1)),
            ("t", ">=", utc(2020, 1, 1)),
        ]

    def test_season_delegates_to_seasonal_filter(self, tmp_path) -> None:
        time_window = window(utc(2020, 1, 1), utc(2020, 12, 31))
        location = ParquetLocation(
            root=str(tmp_path),
            file_name="data.parquet",
            time_window=time_window,
            season=MeteorologicalSeason.DJF,
            time_column="t",
        )
        assert location.time_filter() == _construct_seasonal_filter(
            time_window, time_column="t", season=MeteorologicalSeason.DJF
        )

    @pytest.mark.parametrize("is_aware", [True, False], ids=["aware", "naive"])
    @pytest.mark.parametrize("season", [None, *MeteorologicalSeason])
    def test_matches_brute_force_selection(self, tmp_path, season, is_aware: bool) -> None:
        time_window = window(utc(2020, 1, 15, 18, 30), utc(2021, 12, 31), is_aware=is_aware)
        location = ParquetLocation(
            root=str(tmp_path),
            file_name="data.parquet",
            time_window=time_window,
            season=season,
            time_column="t",
        )
        stamps = stamps_for(time_window)
        expected = brute_force(time_window, season, stamps)
        assert np.array_equal(select(location.time_filter(), stamps), expected)


# Daily through 2020 and 2021, so that each season covers a known number of rows
PARQUET_TIMES: pd.DatetimeIndex = pd.date_range("2020-01-01", "2021-12-31", freq="D")


@pytest.fixture
def parquet_file(tmp_path) -> "Path":
    frame = pd.DataFrame({"time": PARQUET_TIMES, "value": np.arange(len(PARQUET_TIMES), dtype=float)})
    frame.to_parquet(tmp_path / "data.parquet", index=False)
    return tmp_path


@pytest.fixture
def remote_parquet_file():
    fs = fsspec.filesystem("memory")
    root = "/rojak-test-frames"
    frame = pd.DataFrame({"time": PARQUET_TIMES, "value": np.arange(len(PARQUET_TIMES), dtype=float)})
    with fs.open(f"{root}/data.parquet", "wb") as handle:
        frame.to_parquet(handle, index=False)
    yield f"memory://{root}"
    fs.rm(root, recursive=True)


class TestOpenDataframe:
    def test_reads_every_row_without_a_time_window(self, parquet_file) -> None:
        frame = open_dataframe(ParquetLocation(root=str(parquet_file), file_name="data.parquet"))
        assert isinstance(frame, dd.DataFrame)  # lazy, so nothing has been read yet
        assert len(frame.compute()) == len(PARQUET_TIMES)

    def test_pushes_time_window_down_to_the_reader(self, parquet_file) -> None:
        location = ParquetLocation(
            root=str(parquet_file),
            file_name="data.parquet",
            time_window=window(utc(2020, 3, 1), utc(2020, 3, 31), is_aware=False),
            time_column="time",
        )
        times = open_dataframe(location).compute()["time"]
        assert times.min() == pd.Timestamp("2020-03-01")
        assert times.max() == pd.Timestamp("2020-03-31")
        assert len(times) == 31  # noqa: PLR2004

    def test_pushes_season_down_to_the_reader(self, parquet_file) -> None:
        location = ParquetLocation(
            root=str(parquet_file),
            file_name="data.parquet",
            time_window=window(utc(2020, 1, 1), utc(2021, 12, 31), is_aware=False),
            season=MeteorologicalSeason.DJF,
            time_column="time",
        )
        times = open_dataframe(location).compute()["time"]
        expected = PARQUET_TIMES[((PARQUET_TIMES.month % 12 + 3) // 3) == MeteorologicalSeason.DJF.as_int]
        assert sorted(times) == list(expected)

    def test_read_kwargs_are_passed_through(self, parquet_file) -> None:
        frame = open_dataframe(ParquetLocation(root=str(parquet_file), file_name="data.parquet"), columns=["value"])
        assert list(frame.columns) == ["value"]

    def test_read_kwargs_take_precedence_over_defaults(self, parquet_file) -> None:
        location = ParquetLocation(
            root=str(parquet_file),
            file_name="data.parquet",
            time_window=window(utc(2020, 3, 1), utc(2020, 3, 31), is_aware=False),
            time_column="time",
        )
        assert len(open_dataframe(location, filters=None).compute()) == len(PARQUET_TIMES)

    def test_reads_from_a_remote_location(self, remote_parquet_file) -> None:
        location = ParquetLocation(
            root=remote_parquet_file,
            file_name="data.parquet",
            time_window=window(utc(2021, 7, 1), utc(2021, 7, 31), is_aware=False),
            time_column="time",
        )
        times = open_dataframe(location).compute()["time"]
        assert len(times) == 31  # noqa: PLR2004
        assert times.min() == pd.Timestamp("2021-07-01")

    def test_trailing_slash_in_root_does_not_break_the_path(self, parquet_file) -> None:
        location = ParquetLocation(root=f"{parquet_file}/", file_name="data.parquet")
        assert len(open_dataframe(location).compute()) == len(PARQUET_TIMES)
