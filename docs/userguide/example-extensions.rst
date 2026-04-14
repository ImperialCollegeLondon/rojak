Extending ``rojak``
=====================

Other meteorological data sources
-------------------------------------------------

Loading data from a different meteorological data source (i.e. not ERA5 data) can be achieved by,

1. **Adding a new class**: To keep the package structure consistent, create a new module in :py:mod:`rojak.datalib` (e.g., ``rojak/datalib/ifs.py``) and within that create a new class (e.g.  ``IFSData``)
2. **Implementing the interface**: Make the class subclass/implement :py:class:`rojak.core.data.MetData` and satisfy any abstract methods or properties required by that interface.
3. **Providing the adaptor**: Implement the abstract method :py:meth:`MetData.to_clear_air_turbulence_data` on the new class. This method must return a :py:class:`rojak.core.data.CATData` instance.
4. **Verifying compatibility**: Add unit tests that confirm ``to_clear_air_turbulence_data()`` returns a valid ``CATData`` and that required fields are correctly mapped.
5. **Using with diagnostics**: Convert your ``MetData`` instance via :py:meth:`to_clear_air_turbulence_data()` and pass the returned ``CATData`` to ``rojak.turbulence.diagnostic.DiagnosticFactory`` to instantiate diagnostics.
