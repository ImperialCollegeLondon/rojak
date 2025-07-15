import distributed
from dask.base import is_dask_collection


def blocking_wait_futures(dask_collection: object) -> None:
    """
    Utility function to synchronise distributed execution

    Args:
        dask_collection: Dask collection to wait on the futures of. If :py:class:`object` is not a Dask collection,
            nothing happens.

    >>> blocking_wait_futures([])
    >>> from distributed import Client
    >>> client = Client()
    >>> import dask.array as da
    >>> to_persist = da.random.default_rng(5).standard_normal(10).persist()
    >>> blocking_wait_futures(da.random.default_rng(5).standard_normal(10).persist())
    >>> to_persist
    dask.array<standard_normal, shape=(10,), dtype=float64, chunksize=(10,), chunktype=numpy.ndarray>
    >>> client.close()
    """
    if is_dask_collection(dask_collection):
        distributed.wait(distributed.futures_of(dask_collection))
