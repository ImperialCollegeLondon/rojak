Installation Guide
===================

Requirements
-------------------

``rojak`` currently only supports Python >=3.12. This is mainly due to the type hinting syntax that has been used

.. literalinclude:: ../../pyproject.toml
    :start-after: dependencies = [
    :end-before: ]

Installation
-------------------

There are a few way which ``rojak`` can be installed. The easiest is through ``pip``

..
    update this to whatever name is chosen in the ends as rojak is taken on pip

.. code-block::

    $ pip install rojak_cat
    $ uv add rojak # if you use uv as your package manager

..
    Alternatively, it can be installed through Conda_ on the `conda-forge`_ channel

    .. code-block::

        $ conda install -c conda-forge rojak

If you'd like the latest changes, ``rojak`` can be installed from source_

.. code-block::

    $ git clone git@github.com:ImperialCollegeLondon/rojak.git
    $ cd rojak
    $ pip install -e

If you use uv_ as your package manager and have cloned the repo, the core dependencies can be installed through,

.. code-block::

    $ uv sync

If you'd like all the dependencies,

.. code-block::

    $ uv sync --all-groups

If you do not require the docs, dev and test dependencies,

.. code-block::

    $ uv sync --all-groups --no-group test --no-group dev --no-group docs

.. _Conda: https://anaconda.org/anaconda/conda
.. _conda-forge: https://anaconda.org/conda-forge
.. _source: https://github.com/ImperialCollegeLondon/rojak
.. _uv: https://docs.astral.sh/uv/
