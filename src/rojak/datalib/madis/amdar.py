import gzip
import shutil
import tempfile
from pathlib import Path
from typing import FrozenSet, Iterable

import xarray as xr


ALL_AMDAR_DATA_VARS: FrozenSet[str] = frozenset(
    {'nStaticIds', 'staticIds', 'lastRecord', 'invTime', 'prevRecord', 'inventory', 'globalInventory', 'firstOverflow',
     'isOverflow', 'firstInBin', 'lastInBin', 'QCT', 'ICT', 'missingInputMinutes', 'minDate', 'maxDate', 'minSecs',
     'maxSecs', 'latitude', 'latitudeDD', 'latitudeQCA', 'latitudeQCR', 'latitudeQCD', 'longitude', 'longitudeDD',
     'longitudeQCA', 'longitudeQCR', 'longitudeQCD', 'altitude', 'altitudeDD', 'altitudeQCA', 'altitudeQCR',
     'altitudeQCD', 'GPSaltitude', 'baroAltitude', 'timeObs', 'timeObsDD', 'timeObsQCA', 'timeObsQCR', 'timeObsQCD',
     'temperature', 'temperatureDD', 'temperatureQCA', 'temperatureQCR', 'temperatureQCD', 'temperatureICA',
     'temperatureICR', 'windDir', 'windDirDD', 'windDirQCA', 'windDirQCR', 'windDirQCD', 'windSpeed', 'windSpeedDD',
     'windSpeedQCA', 'windSpeedQCR', 'windSpeedQCD', 'heading', 'mach', 'trueAirSpeed', 'trueAirSpeedDD',
     'trueAirSpeedQCA', 'trueAirSpeedQCR', 'trueAirSpeedQCD', 'waterVaporMR', 'downlinkedRH', 'RHfromWVMR',
     'rhUncertainty', 'sensor1RelativeHumidity', 'sensor2RelativeHumidity', 'dewpoint', 'dewpointDD', 'dewpointQCA',
     'dewpointQCR', 'dewpointQCD', 'dewpointICA', 'dewpointICR', 'dewpointUncertainty', 'medTurbulence', 'medEDR',
     'medEDRDD', 'medEDRQCA', 'medEDRQCR', 'medEDRQCD', 'maxTurbulence', 'maxEDR', 'maxEDRDD', 'maxEDRQCA', 'maxEDRQCR',
     'maxEDRQCD', 'turbIndex', 'turbIndexDD', 'turbIndexQCA', 'turbIndexQCR', 'turbIndexQCD', 'timeMaxTurbulence',
     'vertAccel', 'vertGust', 'icingCondition', 'icingConditionDD', 'icingConditionQCA', 'icingConditionQCR',
     'icingConditionQCD', 'en_tailNumber', 'flight', 'tailNumber', 'dataType', 'dataSource', 'dataDescriptor',
     'errorType', 'rollFlag', 'rollQuality', 'waterVaporQC', 'interpolatedTime', 'interpolatedLL', 'tempError',
     'dewpointError', 'windDirError', 'windSpeedError', 'speedError', 'bounceError', 'icingError', 'trueAirSpeedError',
     'turbulenceError', 'correctedFlag', 'rptStation', 'timeReceived', 'fileTimeFSL', 'origAirport', 'orig_airport_id',
     'destAirport', 'dest_airport_id', 'indAltitude', 'relHumidity', 'sounding_flag', 'soundingSecs',
     'sounding_airport_id', 'phaseFlight', 'tamdarCarrier3', 'tamdarCarrier', 'tamdarAcType', 'filterSetNum',
     'wvssTest1'})

class Preprocessor:
    filepaths: Iterable[Path]
    data_vars_for_turbulence: list[str] = ["altitude", "altitudeDD", "bounceError", "correctedFlag", "dataDescriptor",
        "dataSource", "dataType", "dest_airport_id", "en_tailNumber", "heading", "interpolatedLL", "interpolatedTime",
        "latitude", "latitudeDD", "longitude", "longitudeDD", "mach", "maxEDR", "maxEDRDD", "maxTurbulence",
        "medEDR", "medEDRDD", "medTurbulence", "orig_airport_id", "phaseFlight", "speedError", "tempError",
        "temperature", "temperatureDD", "timeMaxTurbulence", "timeObs", "timeObsDD",  "trueAirSpeed",
        "trueAirSpeedDD", "trueAirSpeedError", "turbIndex", "turbIndexDD", "turbulenceError",
        "vertAccel", "vertGust", "windDir", "windDirDD", "windDirError", "windSpeed", "windSpeedDD", "windSpeedError",
    ]
    quality_control_vars: list[str] = ["altitudeDD", "windSpeedDD", "timeObsDD", "latitudeDD", "longitudeDD",
        "maxEDRDD", "medEDRDD", "temperatureDD", "trueAirSpeedDD", "turbIndexDD", "windDirDD", "windSpeedDD"]
    error_vars: list[str] = ["bounceError", "speedError", "turbulenceError",
        ## Including the ones below ends up making a lot of data nan. Leave that data in and let the user decide
        ## what to do with it later
        # "tempError",
        # "trueAirSpeedError",
        # "windDirError",
        # "windSpeedError",
    ]
    dimension_name: str = "recNum"

    def __init__(self, filepaths: Iterable[Path] | Path, glob_pattern: str | None = None):
        if glob_pattern is not None:
            if isinstance(filepaths, Iterable):
                self.filepaths = [path for fpath in filepaths for path in fpath.glob(glob_pattern)]
            else:
                self.filepaths = filepaths.glob(glob_pattern)
        else:
            if isinstance(filepaths, Iterable):
                self.filepaths = filepaths
            else:
                self.filepaths = [filepaths]

    @staticmethod
    def decompress_gz(filepath: Path) -> Path:
        if not filepath.is_file():
            raise FileNotFoundError(filepath)
        elif filepath.suffix != ".gz":
            raise ValueError(f"Unsupported file extension: {filepath.suffix}. File must be .gz")

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path: Path = Path(temp_file.name)

        try:
            with gzip.open(filepath, "rb") as f_in:
                with open(temp_file_path, "wb") as f_out:
                    # noinspection PyTypeChecker
                    shutil.copyfileobj(f_in, f_out)
        except Exception as e:
            temp_file_path.unlink(missing_ok=True)
            raise e

        return temp_file_path

    @staticmethod
    def __mask_invalid_qc_for_var(dataset: xr.Dataset, data_var: str) -> xr.Dataset:
        # Z - Preliminary, no QC
        # C - Coarse pass
        # S - Screened
        # V - Verified
        # G - Good
        return dataset.where(((dataset[data_var] == b"Z") | (dataset[data_var] == b"C") | (dataset[data_var] == b"S") |
                              (dataset[data_var] == b"V") | (dataset[data_var] == b"G")))

    def drop_invalid_qc_data(self, dataset: xr.Dataset) -> xr.Dataset:
        for var in self.quality_control_vars:
            dataset = self.__mask_invalid_qc_for_var(dataset, var)
        return dataset.dropna(self.dimension_name, subset=self.quality_control_vars)

    @staticmethod
    def __mask_invalid_error_var(dataset: xr.Dataset, data_var: str) -> xr.Dataset:
        # value_p:  pass -> char(p) = 112
        # value_-:  unknown: no tests could be performed -> char(-) = 45
        # Filters out value_f:  fail: flagged suspect or bad upon receipt -> var(f) = 102
        return dataset.where(((dataset[data_var] == 112) | (dataset[data_var] == 45)))

    def drop_invalid_error_data(self, dataset: xr.Dataset) -> xr.Dataset:
        for var in self.error_vars:
            dataset = self.__mask_invalid_error_var(dataset, var)
        return dataset.dropna(self.dimension_name, subset=self.error_vars)

    def filter_and_export_as_parquet(self, output_directory: Path):
        for filepath in self.filepaths:
            temp_netcdf_file: Path = self.decompress_gz(filepath)
            set_of_data_vars: set = set(self.data_vars_for_turbulence)
            data: xr.Dataset = xr.open_dataset(temp_netcdf_file, engine="netcdf4", decode_timedelta=True,
                                               drop_variables=ALL_AMDAR_DATA_VARS - set_of_data_vars)
            # Drop all the nan data that's already present in the data
            data = data.dropna(self.dimension_name, subset=["maxEDR", "medEDR", "turbIndex", "medTurbulence", "maxTurbulence"])

            # Make all the data that's invalid based on QC and error nan
            data = self.drop_invalid_qc_data(data)
            data = self.drop_invalid_error_data(data)

            variables_to_keep: list[str] = list(set_of_data_vars - set(self.quality_control_vars) - set(self.error_vars))
            data[variables_to_keep].to_dataframe().to_parquet(output_directory / f"{filepath.stem}.parquet")

            temp_netcdf_file.unlink()





