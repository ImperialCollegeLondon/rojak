API
====

.. currentmodule:: rojak

.. autosummary::
   :toctree: generated/
   :recursive:

    atmosphere
    core
    datalib
    orchestrator
    turbulence

.. Listed explicitly: with ``autosummary_ignore_module_all = False``, recursive autosummary only
   descends into the submodules named in a package's own ``__all__``. ``atmosphere`` has none, so
   it still reaches ``feature_composite``, but ``feature_composite.__all__`` does not name
   ``extension``, so it would not be reached from there without this explicit entry.

.. autosummary::
   :toctree: generated/

    atmosphere.feature_composite.extension

.. toctree::
    references
