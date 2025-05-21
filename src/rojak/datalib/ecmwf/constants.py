from typing import Dict

reanalysis_dataset_names: dict[str, str] = {
    "pressure-level": "reanalysis-era5-pressure-levels",
    "single-level": "reanalysis-era5-single-levels",
}

blank_default: Dict = {
    "product_type": ["reanalysis"],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

cat_data_default: Dict = {
    "product_type": ["reanalysis"],
    "variable": [
        "divergence",
        "geopotential",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "potential_vorticity",
        "vorticity",
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
    "download_format": "unarchived",
}

surface_data_contrails_default: dict = {
    "product_type": ["reanalysis"],
    "variable": [
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "top_net_solar_radiation",
        "top_net_thermal_radiation",
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

# Contains all variables for both CAT and contrails that are on the pressure level
contrail_cat_data_default: dict = {
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
        "vertical_velocity",
    ],
    # From ground to upper aircraft limit for flight optimisation
    "pressure_level": [
        "125",
        "150",
        "175",
        "200",
        "225",
        "250",
        "300",
        "350",
        "400",
        "450",
        "500",
        "550",
        "600",
        "650",
        "700",
        "750",
        "775",
        "800",
        "825",
        "850",
        "875",
        "900",
        "925",
        "950",
        "975",
        "1000",
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
}

data_defaults: Dict[str, Dict] = {
    "cat": cat_data_default,
    "surface": surface_data_contrails_default,
    "contrail": contrail_cat_data_default,
    "blank": blank_default,
}

all_hours: list[str]  = ["00:00", "01:00", "02:00", "03:00", "04:00", "05:00", "06:00", "07:00", "08:00", "09:00", 
                         "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", 
                         "20:00", "21:00", "22:00", "23:00"]  # fmt: skip

six_hourly: list[str] = ["00:00", "06:00", "12:00", "18:00"]  # fmt: skip
