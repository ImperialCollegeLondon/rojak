import distributed
from dask.base import is_dask_collection


def blocking_wait_futures(dask_collection: object) -> None:
    if is_dask_collection(dask_collection):
        distributed.wait(distributed.futures_of(dask_collection))
