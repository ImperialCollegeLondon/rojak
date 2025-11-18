Example Use Cases
===================

Computing Turbulence Percentage
--------------------------------------

This example will enable you to generate plots, like in :numref:`turbulence-percentage`, to explore the climatological distribution of turbulence.

.. _turbulence-percentage:

.. figure:: /_static/multi_diagnostic_f3d_ti1_on_200_light.png
    :align: center

    Probability of encountering light turbulence during the months December, January, February from 2018 to 2024 at 200 hPa for the three-dimensional frontogenesis (F3D) and turbulence index 1 (TI1) diagnostics.

The first step in the process is acquiring the calibration ECMWF ERA5 data from CDS_. 
The ``rojak`` command below requests 6-hourly data from the 1st and 15th of every month from 1980 to 1989 and places it in the folder ``met_data/era5/calibration_data``. 
It uses the default configuration for clear air turbulence (CAT) to specify which variables to request, the product type, which pressure levels and the data format.

.. attention::

    This step will download 42GB of data. Moreover, depending on the CDS_ queue this may several hours.

.. code-block::

    $ rojak data meteorology retrieve -s era5 -y 1980 -y 1981 -y 1982 -y 1983 -y 1984 -y 1985 -y 1986 -y 1987 -y 1988 -y 1989 -m -1 -d 1 -d 15 -n pressure-level --default-name cat -o met_data/era5/calibration_data

.. hint::

    If the above command fails due to the CDS API, check out getting started guide on :ref:`met_data`.

The calibration data is used to compute the threshold values to determine whether light turbulence is present for a given turbulence diagnostic.
The next step is to request the data for the evaluation dataset. In :numref:`turbulence-percentage`, data from the boreal winter (i.e. December, January, and February) of 2018 to 2024 was used.

.. attention::

    This step will download 85GB of data. Moreover, depending on the CDS_ queue this may several hours to a day.

.. code-block::

    $ rojak data meteorology retrieve -s era5 -y 2018 -y 2019 -y 2020 -y 2021 -y 2022 -y 2023 -y 2024 -m 12 -m 1 -m 2 -d -1 -n pressure-level --default-name cat -o met_data/era5/evaluation_data

With these datasets, we can now use ``rojak`` to run the analyses.
:numref:`turbulence-probability-config-yaml` is a ``yaml`` file that will be used to control the analysis performed by ``rojak``.
In this configuration file, 

1. *Line 1*: Defines the configuration for the input data. Here, it only specifies the spatial domain which the analysis is for. In this case, it is for the entire globe.
2. *Line 11*: This is the start of the settings for the CAT analysis. This corresponds to the :py:class:`rojak.orchestrator.configuration.TurbulenceConfig`.
3. *Line 12*: Within the turbulence config, the first setting that needs to be set is the chunking (see Dask docs on :doc:`dask:array-chunks`). This specifies the size of each DASK array chunk. For simplicity, it has been set to the size of the spatial domain
4. *Line 16*: Is a list of the turbulence diagnostics which the analysis is to be performed on. Each item in the list must be an string from the :py:class:`enum.StrEnum` class :py:class:`rojak.orchestrator.configuration.TurbulenceDiagnostics`
5. *Line 19*: Defines which phases to run and corresponds to the class :py:class:`rojak.orchestrator.configuration.TurbulencePhases`
6. *Line 20*: Specifies what should occur during the calibration phase. If applicable, where the calibration data is stored. This corresponds to the :py:class:`rojak.orchestrator.configuration.TurbulenceCalibrationPhases` class.
7. *Line 21*: Contains the required configuration for the calibration phase, such as where the data is stored in ``calibration_data_dir``. This corresponds to the :py:class:`rojak.orchestrator.configuration.TurbulenceCalibrationConfig`. As the thresholds phase is specified in line 29, the percentile thresholds for a given turbulence intensity needs to be provided.
8. *Line 29*: Specifies which phases to run during the calibration phase this corresponds to the :py:class:`enum.StrEnum` :py:class:`rojak.orchestrator.configuration.TurbulenceCalibrationPhaseOption`
9. *Line 32*: Contains the required configuration for the evaluation phase, this corresponds to the :py:class:`rojak.orchestrator.configuration.TurbulenceEvaluationConfig`
10. *Line 33*: Specifies which phases to run during the evaluation phase this corresponds to the :py:class:`enum.StrEnum` :py:class:`rojak.orchestrator.configuration.TurbulenceEvaluationPhaseOption`. As ``"probabilities"`` has been specified, it will use the percentile thresholds computed threshold for the turbulence diagnostics during the calibration phase. The option ``"edr"`` will perform the mapping to EDR values using the distribution computed during the calibration phase.


.. code-block:: yaml
    :linenos:
    :caption: turbulence-probability-config.yaml
    :name: turbulence-probability-config-yaml

    data_config:
        spatial_domain:
            maximum_latitude: 90.0
            maximum_longitude: 180.0
            minimum_latitude: -90.0
            minimum_longitude: -180.0
    image_format: eps
    name: eighties
    output_dir: output
    plots_dir: plots
    turbulence_config:
        chunks:
            pressure_level: 3
            latitude: 721
            longitude: 1440
        diagnostics:
            - f3d
            - ti1
        phases:
            calibration_phases:
                calibration_config:
                    calibration_data_dir: met_data/era5/calibration_data
                    percentile_thresholds:
                        light: 97.0
                        light_to_moderate: 99.1
                        moderate: 99.6
                        moderate_to_severe: 99.8
                        severe: 99.9
                phases:
                    - thresholds
                    - histogram
            evaluation_phases:
                phases:
                    - probabilities
                    - edr
                evaluation_config:
                    evaluation_data_dir: met_data/era5/evaluation_data

This configuration can be launched using the command below,

.. code-block::

    $ rojak run turbulence-probability-config.yaml

To monitor the progress of the process through the Dask :doc:`dask:dashboard`, go to `http://localhost:8787/status <http://localhost:8787/status>`_.

.. note::

    This configuration was run on the HPC with 920GB of memory.
    It is possible that it does not require as much. It is likely to require a minimum of 42GB of memory.
    Moreover, as the default behaviour of dask (see :doc:`dask:configuration`) is to write to ``/tmp`` its ability to spill to disk may be limited to the amount of memory in your system.


EDR Snapshot
--------------------------------------

By default, ``rojak`` does not produce snapshot plots. This can be achieved through a few different ways.
The method shown below is the least involved, thus has the least room for customisation.

.. image:: /_static/multi_edr_f3d_ti1.png

.. code-block:: yaml
    :linenos:
    :caption: edr-snapshot-config.yaml
    :name: edr-snapshot-config-yaml

    data_config:
        spatial_domain:
            maximum_latitude: 70
            maximum_longitude: 60
            minimum_latitude: 0
            minimum_longitude: -150
    image_format: png
    name: eighties
    output_dir: output
    plots_dir: plots
    turbulence_config:
        chunks:
            pressure_level: 3
            latitude: 721
            longitude: 1440
        diagnostics:
            - f3d
            - ti1
        phases:
            calibration_phases:
                calibration_config:
                    calibration_data_dir: met_data/era5/calibration_data
                phases:
                    - histogram
            evaluation_phases:
                phases:
                    - edr
                evaluation_config:
                    evaluation_data_dir: met_data/era5/evaluation_data


.. literalinclude:: edr-snapshot.py
    :language: python
    :linenos:
    :caption: edr-snapshot.py
    :name: edr-snapshot.py


.. _CDS: https://cds.climate.copernicus.eu/