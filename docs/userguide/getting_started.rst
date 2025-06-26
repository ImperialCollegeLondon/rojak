
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

1. ``run``: Performing the turbulence analysis on data. As there are a multitude of permutations which would involve
   too many CLI flags, the control flow of the program is dictated by the configuration file. The full set of available
   settings can be found in the :doc:`configuration </api/generated/rojak.orchestrator.configuration>` API reference
2. ``data``: Command for retrieving and pre-processing turbulence data

Computing Clear Air Turbulence (CAT) diagnostics
-------------------------------------------------


