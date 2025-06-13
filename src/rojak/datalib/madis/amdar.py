import fnmatch
import gzip
import shutil
import tempfile
from ftplib import FTP
from pathlib import Path
from typing import FrozenSet, Iterable

import dask.dataframe as dd
import xarray as xr
from rich.progress import track

from rojak.core.data import AmdarData, DataPreprocessor, DataRetriever, Date

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
     'wvssTest1'})  # fmt: skip


class MadisAmdarPreprocessor(DataPreprocessor):
    filepaths: list[Path]
    data_vars_for_turbulence: set[str] = {"altitude", "altitudeDD", "bounceError", "correctedFlag", "dataDescriptor",
        "dataSource", "dataType", "dest_airport_id", "en_tailNumber", "heading", "interpolatedLL", "interpolatedTime",
        "latitude", "latitudeDD", "longitude", "longitudeDD", "mach", "maxEDR", "maxEDRDD", "maxTurbulence",
        "medEDR", "medEDRDD", "medTurbulence", "orig_airport_id", "phaseFlight", "speedError", "tempError",
        "temperature", "temperatureDD", "timeMaxTurbulence", "timeObs", "timeObsDD",  "trueAirSpeed",
        "trueAirSpeedDD", "trueAirSpeedError", "turbIndex", "turbIndexDD", "turbulenceError",
        "vertAccel", "vertGust", "windDir", "windDirDD", "windDirError", "windSpeed", "windSpeedDD", "windSpeedError",
    }  # fmt: skip
    quality_control_vars: set[str] = {"altitudeDD", "windSpeedDD", "timeObsDD", "latitudeDD", "longitudeDD",
        "maxEDRDD", "medEDRDD", "temperatureDD", "trueAirSpeedDD", "turbIndexDD", "windDirDD"}  # fmt: skip
    error_vars: set[str] = {"bounceError", "speedError", "turbulenceError",
        ## Including the ones below ends up making a lot of data nan. Leave that data in and let the user decide
        ## what to do with it later
        # "tempError",
        # "trueAirSpeedError",
        # "windDirError",
        # "windSpeedError",
    }  # fmt: skip
    dimension_name: str = "recNum"
    relative_to_root_path: list[Path] | None = None

    def __init__(self, filepaths: Iterable[Path] | Path, glob_pattern: str | None = None) -> None:
        if glob_pattern is not None:
            target_files: list[Path] = []
            self.relative_to_root_path = []
            if isinstance(filepaths, Iterable):
                for base_fpath in filepaths:
                    for path in base_fpath.glob(glob_pattern):
                        target_files.append(path)
                        self.relative_to_root_path.append(path.relative_to(base_fpath).parents[0])
            else:
                self.relative_to_root_path = []
                for item in filepaths.glob(glob_pattern):
                    target_files.append(item)
                    self.relative_to_root_path.append(item.relative_to(filepaths).parents[0])

            assert len(target_files) > 0
            self.filepaths = target_files
        else:
            self.relative_to_root_path = None
            if isinstance(filepaths, Iterable):
                self.filepaths = list(filepaths)
            else:
                if not filepaths.is_file():
                    raise ValueError(
                        f"File {filepaths} is not a file. As glob pattern is not defined this must be a file"
                    )
                self.filepaths = [filepaths]

    @staticmethod
    def decompress_gz(filepath: Path) -> Path:
        if not filepath.is_file():
            raise FileNotFoundError(filepath)
        if filepath.suffix != ".gz":
            raise ValueError(f"Unsupported file extension: {filepath.suffix}. File must be .gz")

        temp_file_path: Path

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path: Path = Path(temp_file.name)

            try:
                with gzip.open(filepath, "rb") as f_in, temp_file_path.open(mode="wb") as f_out:
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
        return dataset.where(
            (
                (dataset[data_var] == b"Z")
                | (dataset[data_var] == b"C")
                | (dataset[data_var] == b"S")
                | (dataset[data_var] == b"V")
                | (dataset[data_var] == b"G")
            )
        )

    def drop_invalid_qc_data(self, dataset: xr.Dataset) -> xr.Dataset:
        qc_vars_present: set[str] = dataset.data_vars.keys() & self.quality_control_vars
        for var in qc_vars_present:
            dataset = self.__mask_invalid_qc_for_var(dataset, var)
        return dataset.dropna(self.dimension_name, subset=qc_vars_present)

    @staticmethod
    def __mask_invalid_error_var(dataset: xr.Dataset, data_var: str) -> xr.Dataset:
        # value_p:  pass -> char(p) = 112
        # value_-:  unknown: no tests could be performed -> char(-) = 45
        # Filters out value_f:  fail: flagged suspect or bad upon receipt -> var(f) = 102
        return dataset.where(((dataset[data_var] == ord("p")) | (dataset[data_var] == ord("-"))))

    def drop_invalid_error_data(self, dataset: xr.Dataset) -> xr.Dataset:
        error_vars_present: set[str] = dataset.data_vars.keys() & self.error_vars
        for var in error_vars_present:
            dataset = self.__mask_invalid_error_var(dataset, var)
        return dataset.dropna(self.dimension_name, subset=error_vars_present)

    def apply_preprocessor(self, output_directory: Path) -> None:
        # Filters and exports data to parquet
        output_directory.mkdir(parents=True, exist_ok=True)

        for index, filepath in track(enumerate(self.filepaths), total=len(self.filepaths)):
            temp_netcdf_file: Path = self.decompress_gz(filepath)
            data: xr.Dataset = xr.open_dataset(
                temp_netcdf_file,
                engine="netcdf4",
                decode_timedelta=True,
                drop_variables=ALL_AMDAR_DATA_VARS - self.data_vars_for_turbulence,
            )

            turbulence_subset: set[str] = data.data_vars.keys() & {
                "maxEDR",
                "medEDR",
                "turbIndex",
                "medTurbulence",
                "maxTurbulence",
            }
            # Drop all the nan data that's already present in the data
            data = data.dropna(self.dimension_name, subset=turbulence_subset)

            # Make all the data that's invalid based on QC and error nan
            data = self.drop_invalid_qc_data(data)
            data = self.drop_invalid_error_data(data)

            variables_to_keep: list[str] = list(
                (data.data_vars.keys() & self.data_vars_for_turbulence) - self.quality_control_vars - self.error_vars
            )
            output_file: Path = (
                output_directory / f"{filepath.stem}.parquet"
                if self.relative_to_root_path is None
                else output_directory / self.relative_to_root_path[index] / f"{filepath.stem}.parquet"
            )  # fmt: skip
            output_file.parent.mkdir(parents=True, exist_ok=True)
            dd.from_pandas(data[variables_to_keep].to_dataframe()).to_parquet(output_file)

            temp_netcdf_file.unlink()
            del data


class AcarsRetriever(DataRetriever):
    ftp_host: str = "madis-data.ncep.noaa.gov"
    product: str = "acars"
    file_pattern: str

    def __init__(self, file_pattern: str | None = None) -> None:
        if file_pattern is None:
            self.file_pattern = "*.gz"
        else:
            self.file_pattern = file_pattern

    def _download_file(self, date: Date, base_output_dir: Path) -> None:
        output_dir: Path = (base_output_dir / f"{date.year:02d}" / f"{date.month:02d}").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with FTP(self.ftp_host) as ftp:
            ftp.login()
            ftp.cwd(f"archive/{date.year}/{date.month:02d}/{date.day:02d}/point/{self.product}/netcdf/")
            files: list[str] = ftp.nlst()
            matching_files: list[str] = fnmatch.filter(files, self.file_pattern)
            for file in matching_files:
                target_file_path: Path = output_dir / file
                with target_file_path.open(mode="wb") as f_out:
                    ftp.retrbinary(f"RETR {file}", f_out.write)

    def download_files(
        self,
        years: list[int],
        months: list[int],
        days: list[int],
        base_output_dir: Path,
    ) -> None:
        dates: list[Date] = self.compute_date_combinations(years, months, days)
        for date in track(dates):
            self._download_file(date, base_output_dir)


class AcarsAmdarData(AmdarData):
    def __init__(self, path_to_files: str | list) -> None:
        super().__init__(path_to_files, "baroAltitude")

    def load(self) -> "dd.DataFrame":
        return dd.read_parquet(self._path_to_files)


def load_acars_amdar_data(path: str | list) -> "dd.DataFrame":
    # ASSUMES FILES ARE IN PARQUET FORMAT
    return dd.read_parquet(path, filesystem="arrow")
