
Getting Started
===================

What in the World Is ``rojak``? What Can I Use It For?
-------------------------------------------------------

Rojak is salad with Javanese origins. Colloquially (in Malaysia and Singapore), it means a mixture of things.
``rojak`` is a Python library and command line interface (CLI) tool for analysing aviation turbulence.

Command Line Interface
------------------------

Once ``rojak`` has been installed, the commands of the CLI tool is displayed via,

.. code-block::

    $ rojak --help

     Usage: rojak [OPTIONS] COMMAND [ARGS]...

    ╭─ Options ────────────────────────────────────────────────────────────────────╮
    │ --install-completion          Install completion for the current shell.      │
    │ --show-completion             Show completion for the current shell, to copy │
    │                               it or customize the installation.              │
    │ --help                        Show this message and exit.                    │
    ╰──────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ───────────────────────────────────────────────────────────────────╮
    │ run    Performs analysis on data according to configuration file             │
    │ data   Perform operations on data                                            │
    ╰──────────────────────────────────────────────────────────────────────────────╯

The ``--help`` can be run on the commands (e.g. ``run``) to information about the required arguments.

To install the auto-completions for your shell,

.. code-block::

    $ rojak --install-completion

``rojak`` has several functionalities. As such, each functionality corresponds to a command within the CLI.
Currently, there two main functions,

#. ``run``: Performing the turbulence analysis on data. As there are a multitude of permutations which would involve too many CLI flags, the control flow of the program is dictated by the configuration file. The full set of available settings can be found in the :doc:`configuration </api/generated/rojak.orchestrator.configuration>` API reference
#. ``data``: Command for retrieving and pre-processing turbulence data

Computing Clear Air Turbulence (CAT) diagnostics
-------------------------------------------------

The analysis through the CLI tool is controlled through a yaml configuration file. Below is an example ``yaml`` file.

.. code-block:: yaml

    contrails_config: null
    data_config:
      meteorology_config: null
      spatial_domain:
        maximum_latitude: 90.0
        maximum_longitude: 180.0
        minimum_latitude: -90.0
        minimum_longitude: -180.0
    image_format: png
    name: test
    output_dir: output
    plots_dir: plots
    turbulence_config:
      chunks:
        pressure_level: 3
        latitude: 721
        longitude: 1440
      diagnostics:
        - deformation
        - richardson
        - f3d
        - ubf
        - ti1
        - ngm1
        - nva
        - dutton
        - edr_lunnon
        - brown1
      phases:
        calibration_phases:
          calibration_config:
            calibration_data_dir: <path_to_meteorological_calibration_data>
            diagnostic_distribution_file_path: null
            percentile_thresholds:
              light: 97.0
              light_to_moderate: 99.1
              moderate: 99.6
              moderate_to_severe: 99.8
              severe: 99.9
            thresholds_file_path: null
          phases:
            - thresholds
            - histogram
        evaluation_phases:
          phases:
            - probabilities
            - edr
            - correlation_between_probabilities
            - correlation_between_edr
          evaluation_config:
            evaluation_data_dir: <path_to_meteorological_evaluation_data>

.. code-block::

    $ rojak run <path_to_config_file>.yaml

Running ``rojak`` with the above command and configuration file would,

#. Use data from the entire globe as specified in ``data_config.spatial_domain``
#. Output any plots in ``png`` (``image_format``) into the ``plots/`` directory (``plots_dir``)
#. For the turbulence computations,

   #. Arrays will be chunked according to the dictionary in ``turbulence_config.chunks``. It is important that the chunks are large enough such that derivatives in the 3 spatial dimensions are valid.
   #. The turbulence diagnostics listed under ``turbulence_config.diagnostics`` will be computed, e.g. the Richardson number (``richardson``) and the three-dimensions frontogenesis equation (``f3d``)

#. The turbulence analyses require two set of data - calibration and evaluation. The calibration phase is used to determine the threshold values for determining the turbulence intensity and the distribution of diagnostics to map it to EDR. In the example configuration, ``turbulence_config.phases.calibration_phases.phases`` specifies two phases: thresholds and histogram. Since the threshold phase is specified, the percentiles used to compute the thresholds must be too.
#. The evaluation phases specified will use the output of the calibration phases

For more details on what can be specified and the various options for the configuration file, see :py:mod:`rojak.orchestrator.configuration`.


Retrieving Data
-------------------------------------------------

``rojak``'s ``data`` command in the CLI tool allows one to perform operations on data,

.. code-block::

    $ rojak data --help

     Usage: rojak data [OPTIONS] COMMAND [ARGS]...

     Perform operations on data


    ╭─ Options ────────────────────────────────────────────────────────────────────────────────╮
    │ --help          Show this message and exit.                                              │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Commands ───────────────────────────────────────────────────────────────────────────────╮
    │ amdar         Operations for AMDAR data                                                  │
    │ meteorology   Operations for Meteorology data                                            │
    ╰──────────────────────────────────────────────────────────────────────────────────────────╯

As there are two main types of data, the operations that can be performed are split based on whether it is AMDAR data or meteorological data.

AMDAR Data
^^^^^^^^^^^^^^^^^

``rojak`` has two commands for AMDAR data: ``retrieve`` and ``preprocess``. The ``retrieve`` command gets the data from a public server. As `UKMO MetDB AMDAR data <https://catalogue.ceda.ac.uk/uuid/33f44351f9ceb09c495b8cef74860726>`__ is not publicly available, it is not currently supported.

``rojak`` retrieves `AMDAR data from the NOAA MADIS distribution service <https://amdar.ncep.noaa.gov/index.shtml>`__. Currently, it only supports retrieving data from the `public ftp server <https://madis-data.ncep.noaa.gov/index.shtml>`__.

Running the ``--help`` with the command shows the possible arguments,

.. code-block::

    $ rojak data amdar retrieve --help

For example, if you would like to download all the data in 2024 and place them in the ``~/noaa_amdar`` folder, the command would be,

.. code-block::

    $ rojak data amdar retrieve -s madis -y 2024 -m -1 -d -1 -o ~/noaa_amdar

This allows the user to specify what date range the data should be retrieved. For ``rojak`` to use ACARS AMDAR data in its turbulence analysis, it must first be preprocessed and converted to `parquet <https://parquet.apache.org/>`__ files. To preprocess the data retrieved in the previous step, the command would be,

.. code-block::

    $ rojak data amdar preprocess -s madis -i ~/noaa_amdar -o ~/processed_noaa_amdar

To use this processed AMDAR data in the turbulence analysis, the :py:class:`rojak.orchestrator.configuration.AmdarConfig` to the :py:class:`rojak.orchestrator.configuration.DataConfig` in the configuration file. For example,

.. code-block:: yaml

    data_config:
      spatial_domain:
        maximum_latitude: 90.0
        maximum_longitude: 180.0
        minimum_latitude: -90.0
        minimum_longitude: -180.0
      amdar_config:
        data_dir: "~/processed_noaa_amdar/"
        glob_pattern: "**/*.parquet"
        data_source: madis
        time_window:
          lower: "2024-01-01"
          upper: "2024-12-12T18:00"

Meteorological Data: ERA5
^^^^^^^^^^^^^^^^^^^^^^^^^^

``rojak`` uses the `Climate Data Store (CDS) <https://cds.climate.copernicus.eu/>`__'s API to retrieve the ERA5 data. To use the functionality, you will need to follow the `CDS instructions on setup to use the API <https://cds.climate.copernicus.eu/how-to-api>`__. The optional dependency in the ``ecmwf`` group must be installed.

Once everything has been setup, you could, for example, download all the data for 2024 on pressure levels for CAT diagnostics using,

.. code-block::

    $ rojak data meteorology retrieve -s era5 -y 2024 -m -1 -d -1 --data-set-name "pressure-level" --default-name cat
