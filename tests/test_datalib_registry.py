import pytest

from rojak.datalib import registry
from rojak.datalib.ecmwf.era5 import Era5Data
from rojak.orchestrator.configuration import MetDataSource, MeteorologyConfig, SpatialDomain


def test_era5_registered_by_default() -> None:
    assert registry.get_met_data_adapter(MetDataSource.ERA5) is Era5Data


def test_unregistered_source_raises(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_MET_DATA_ADAPTERS", {})
    with pytest.raises(NotImplementedError):
        registry.get_met_data_adapter(MetDataSource.ERA5)


def test_register_refuses_to_overwrite_unless_asked(monkeypatch) -> None:
    monkeypatch.setattr(registry, "_MET_DATA_ADAPTERS", dict(registry._MET_DATA_ADAPTERS))
    sentinel = object()
    with pytest.raises(ValueError, match="already registered"):
        registry.register_met_data_adapter(MetDataSource.ERA5, lambda _: sentinel)  # pyright: ignore[reportArgumentType]
    registry.register_met_data_adapter(MetDataSource.ERA5, lambda _: sentinel, overwrite=True)  # pyright: ignore[reportArgumentType]
    assert registry.get_met_data_adapter(MetDataSource.ERA5)(None) is sentinel  # pyright: ignore[reportArgumentType]


def test_load_cat_data_dispatches_through_registry(mocker, tmp_path) -> None:
    config = MeteorologyConfig(location={"type": "files", "root": str(tmp_path)})  # pyright: ignore[reportArgumentType]
    domain = SpatialDomain(minimum_latitude=0, maximum_latitude=10, minimum_longitude=0, maximum_longitude=10)
    open_mock = mocker.patch.object(registry.sources, "open_dataset", return_value="raw dataset")
    adapter = mocker.Mock()
    mocker.patch.dict(registry._MET_DATA_ADAPTERS, {MetDataSource.ERA5: adapter})

    result = registry.load_cat_data(config, domain, {"time": 1})

    open_mock.assert_called_once_with(config.location, chunks={"time": 1}, engine="h5netcdf")
    adapter.assert_called_once_with("raw dataset")
    adapter.return_value.to_clear_air_turbulence_data.assert_called_once_with(domain)
    assert result is adapter.return_value.to_clear_air_turbulence_data.return_value
