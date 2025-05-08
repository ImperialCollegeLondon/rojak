from typing import Dict

DATA_SET_NAME: str = "reanalysis-era5-pressure-levels"
CAT_DATA_DEFAULT: Dict = {
    "product_type": [
        "reanalysis"
    ],
    "variable": [
        "divergence",
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "potential_vorticity",
        "vorticity"
    ],
    # Needs year, month, day, time to be specified
    "pressure_level": [
        "175",
        "200",  # Williams focuses on 200hPA
        "225",
    ],
    # More pressure levels for contrail turbulence relation
    # "pressure_level": [
    #    "150", "175", "200",
    #    "225", "250", "300",
    #    "350", "400", "450",
    #    "500"
    # ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

SINGLE_LEVEL_DATASET_NAME: str = "reanalysis-era5-single-levels"
SURFACE_DATA_CONTRAILS_DEFAULT: dict = {
    "product_type": ["reanalysis"],
    "variable": [
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "top_net_solar_radiation",
        "top_net_thermal_radiation"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}

# Contains all variables for both CAT and contrails that are on the pressure level
CONTRAIL_CAT_DATA_DEFAULT: dict = {
    "product_type": ["reanalysis"],
    "variable": [
        "divergence",
        "fraction_of_cloud_cover",
        "geopotential",
        "potential_vorticity",
        "relative_humidity",
        "specific_cloud_ice_water_content",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity"
    ],
    # From ground to upper aircraft limit for flight optimisation
    "pressure_level": [
        "125", "150", "175",
        "200", "225", "250",
        "300", "350", "400",
        "450", "500", "550",
        "600", "650", "700",
        "750", "775", "800",
        "825", "850", "875",
        "900", "925", "950",
        "975", "1000"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived"
}