#  Copyright (c) 2025-present Hui Ling Wong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""
Describe *where* gridded input data lives, and open it as a single :class:`xarray.Dataset`

This module only answers "which files/stores make up this dataset, and how are they opened". Adapting the opened
dataset into rojak's variables (e.g. :class:`~rojak.core.data.CATData`) is the job of a
:class:`~rojak.core.data.MetData` subclass, selected via :mod:`rojak.datalib.registry`.

A location is one of the following, distinguished by its ``type`` key in the configuration (a pydantic
discriminated union, see :data:`DataLocation`):

- :class:`FilesLocation` (``type: files``): every file matching a glob pattern under a root. This is the
  behaviour rojak has always had (``data_dir`` + ``glob_pattern``).
- :class:`TemplateLocation` (``type: template``): files named by a :meth:`str.format` template over a time window,
  e.g. ``"{time:%Y}/{time:%m}/{time:%Y-%m-%d}.nc"``. Selects data by *time* rather than by folder layout, in the
  spirit of satpy/trollsift, ESMValTool's ``input_dir`` templates and Pangeo Forge's ``FilePattern``.
- :class:`ZarrLocation` (``type: zarr``): a single (local or remote) zarr store.

Every ``root``/``store`` is an fsspec URL path, so a plain local path (``~/data/era5``), an explicit ``file://``
URL and remote URLs (``s3://``, ``gs://``, ``https://``, ``memory://``, ...) are all handled the same way. Remote
protocols need their fsspec implementation installed (e.g. ``s3fs``, ``gcsfs``) and can be given credentials or
other options via ``storage_options``.
"""

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, assert_never

import fsspec
import pandas as pd
import xarray as xr
from fsspec.utils import get_protocol
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rojak.utilities.types import Limits

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fsspec import AbstractFileSystem

logger = logging.getLogger(__name__)

type NetcdfEngine = Literal["netcdf4", "scipy", "pydap", "h5netcdf"]

_LOCAL_PROTOCOLS: frozenset[str] = frozenset({"file", "local"})


def is_remote(urlpath: str | Path) -> bool:
    """
    Whether ``urlpath`` refers to a non-local filesystem

    Args:
        urlpath: Local path or fsspec URL

    Returns:
        ``True`` if the URL's protocol is anything other than the local filesystem

    Examples:
        >>> is_remote("~/data/era5")
        False
        >>> is_remote("file:///data/era5")
        False
        >>> is_remote("s3://bucket/era5")
        True
    """
    return get_protocol(str(urlpath)) not in _LOCAL_PROTOCOLS


def _filesystem(urlpath: str, storage_options: "Mapping[str, Any]") -> tuple["AbstractFileSystem", str]:
    """
    Resolve ``urlpath`` into its fsspec filesystem and the protocol-less path on that filesystem

    Local paths have ``~`` expanded, as fsspec does not do so itself
    """
    if not is_remote(urlpath):
        urlpath = str(Path(fsspec.core.strip_protocol(urlpath)).expanduser())
    return fsspec.core.url_to_fs(urlpath, **storage_options)


def _to_openable(fs: "AbstractFileSystem", path: str) -> str:
    """Path as returned by ``fs`` turned back into something xarray/fsspec can open (re-attaching the protocol)"""
    return path if get_protocol(fs.unstrip_protocol(path)) in _LOCAL_PROTOCOLS else fs.unstrip_protocol(path)


class _BaseLocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Options passed to the fsspec filesystem (e.g. credentials, anonymous access)",
        repr=False,
    )

    def _check_local_root_is_dir(self, root: str) -> None:
        if not is_remote(root):
            fs, path = _filesystem(root, self.storage_options)
            if not fs.isdir(path):
                raise ValueError(f"Data directory {root} does not exist or is not a directory")


class FilesLocation(_BaseLocation):
    """Every file under :attr:`root` matching :attr:`glob_pattern`"""

    type: Literal["files"] = "files"
    root: str = Field(description="Directory (local path or fsspec URL) containing the data files", repr=True)
    glob_pattern: str = Field(
        default="*.nc", description="Glob pattern, relative to root, matching the data files", repr=True
    )

    @model_validator(mode="after")
    def check_root_is_dir(self) -> Self:
        """Validate that a local :attr:`root` is an existing directory (remote roots are not checked eagerly)"""
        self._check_local_root_is_dir(self.root)
        return self


class TemplateLocation(_BaseLocation):
    """
    Files under :attr:`root` named by :attr:`template` formatted with each time step in :attr:`time_window`

    Example:
        ``template="{time:%Y}/{time:%m}/{time:%Y-%m-%d}.nc"`` with a daily :attr:`frequency` over January 2024
        resolves to ``2024/01/2024-01-01.nc`` ... ``2024/01/2024-01-31.nc``. Time steps that format to the same
        name (e.g. a monthly-file template with an hourly frequency) are de-duplicated.
    """

    type: Literal["template"] = "template"
    root: str = Field(description="Directory (local path or fsspec URL) the template is relative to", repr=True)
    template: str = Field(description="str.format template for each file name, using the `time` field", repr=True)
    time_window: Limits[datetime.datetime] = Field(description="Inclusive time window to load", repr=True)
    frequency: str = Field(default="1D", description="pandas frequency string to step through time_window")
    on_missing: Literal["raise", "skip"] = Field(
        default="raise", description="What to do if a file produced by the template does not exist"
    )

    @model_validator(mode="after")
    def check_template_and_window(self) -> Self:
        """Validate that :attr:`template` uses ``time``, :attr:`time_window` is increasing, and a local root exists"""
        if "{time" not in self.template:
            raise ValueError("Template must contain a `{time...}` field")
        if self.time_window.lower > self.time_window.upper:
            raise ValueError("Time window must be increasing from lower to upper")
        self._check_local_root_is_dir(self.root)
        return self

    def time_steps(self) -> pd.DatetimeIndex:
        """Time steps in :attr:`time_window` (inclusive) at :attr:`frequency`"""
        return pd.date_range(self.time_window.lower, self.time_window.upper, freq=self.frequency)


class ZarrLocation(_BaseLocation):
    """A single zarr store"""

    type: Literal["zarr"] = "zarr"
    store: str = Field(description="Zarr store (local path or fsspec URL)", repr=True)
    consolidated: bool | None = Field(default=None, description="Whether the store has consolidated metadata")


DataLocation = Annotated[FilesLocation | TemplateLocation | ZarrLocation, Field(discriminator="type")]
"""Where a gridded dataset lives, see the module docstring for the options"""


def resolve_paths(location: FilesLocation | TemplateLocation) -> list[str]:
    """
    Resolve a file-based location into the (sorted, for :class:`FilesLocation`) list of files to open

    Args:
        location: Location to resolve

    Returns:
        Local paths, or protocol-qualified URLs for remote filesystems

    Raises:
        FileNotFoundError: If no files are found, or (with ``on_missing="raise"``) a templated file is missing
    """
    fs, root = _filesystem(location.root, location.storage_options)
    root = root.rstrip("/")
    match location:
        case FilesLocation():
            found: list[str] = sorted(str(path) for path in fs.glob(f"{root}/{location.glob_pattern}"))
        case TemplateLocation():
            candidates: list[str] = list(
                dict.fromkeys(f"{root}/{location.template.format(time=step)}" for step in location.time_steps())
            )
            found = [path for path in candidates if fs.exists(path)]
            missing: list[str] = sorted(set(candidates) - set(found))
            if missing and location.on_missing == "raise":
                raise FileNotFoundError(f"{len(missing)} templated file(s) not found, first is {missing[0]}")
            if missing:
                logger.warning("Skipping %d missing templated file(s)", len(missing))
        case _ as unreachable:
            assert_never(unreachable)

    if not found:
        raise FileNotFoundError(f"No data files found for {location!r}")
    return [_to_openable(fs, path) for path in found]


def open_dataset(
    location: FilesLocation | TemplateLocation | ZarrLocation,
    chunks: "Mapping | None",
    engine: NetcdfEngine = "h5netcdf",
    is_decoded: bool = True,
) -> xr.Dataset:
    """
    Lazily open ``location`` as a single dataset

    File-based locations are concatenated with :func:`xarray.open_mfdataset`. Remote files are opened through
    fsspec as file-like objects, which only the ``h5netcdf`` engine supports.

    Args:
        location: Where the data lives
        chunks: Dask chunking to use. Must not be ``None`` for file-based locations.
        engine: Backend engine used to read netCDF files (ignored for zarr). Defaults to ``"h5netcdf"``.
        is_decoded: Whether to decode coordinates and CF conventions on load. Defaults to ``True``.

    Returns:
        Lazily loaded dataset

    Raises:
        ValueError: If ``chunks`` is ``None`` for a file-based location, or a remote location is opened with an
            engine other than ``h5netcdf``
        FileNotFoundError: See :func:`resolve_paths`
    """
    match location:
        case ZarrLocation():
            return xr.open_zarr(
                location.store,
                # pyright takes issue with the chunks kwarg. This is due to it being untyped
                # See https://github.com/pydata/xarray/issues/11221
                chunks=chunks if chunks is not None else "auto",  # pyright: ignore[reportArgumentType]
                storage_options=location.storage_options or None,
                consolidated=location.consolidated,
                decode_coords=is_decoded,
                decode_cf=is_decoded,
            )
        case FilesLocation() | TemplateLocation():
            if chunks is None:
                raise ValueError("Chunks for multi-file load cannot be None")
            paths: list[str] = resolve_paths(location)
            logger.debug("Opening %d files from %s", len(paths), location.root)
            if is_remote(location.root):
                if engine != "h5netcdf":
                    raise ValueError(f"Remote netCDF files can only be opened with h5netcdf, not {engine}")
                to_open: list = [
                    file.open() for file in fsspec.open_files(paths, mode="rb", **location.storage_options)
                ]
            else:
                to_open = paths
            return xr.open_mfdataset(
                to_open,
                chunks=chunks,
                parallel=True,
                engine=engine,
                decode_coords=is_decoded,
                decode_cf=is_decoded,
                decode_timedelta=True,
                # xarray >= 2025.08 refuses compat="override" with the default coords="different" once more
                # than one file is concatenated. xarray's annotation does not admit the literal it documents
                coords="minimal",  # pyright: ignore[reportArgumentType]
                compat="override",
            )
        case _ as unreachable:
            assert_never(unreachable)
