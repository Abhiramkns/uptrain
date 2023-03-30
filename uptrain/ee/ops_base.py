"""Operators for the uptrain.core module. 

- Arrow/Numpy conversion utils - since we leverage duckdb as a cache and ray for execution, intermediate 
outputs are stored as Arrow batches. 
"""

from typing import Union, Protocol, runtime_checkable

import numpy as np
import pyarrow as pa

# -----------------------------------------------------------
# utility routines for converting between Arrow and Numpy
# -----------------------------------------------------------


def array_np_to_arrow(arr: np.ndarray) -> pa.Array:
    assert arr.ndim in (1, 2), "Only 1D and 2D arrays are supported."
    if arr.ndim == 1:
        return pa.array(arr)
    else:
        dim1, dim2 = arr.shape
        return pa.ListArray.from_arrays(
            np.arange(0, (dim1 + 1) * dim2, dim2), arr.ravel()
        )


def array_arrow_to_np(arr: pa.Array) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    if not pa.types.is_list(arr.type):
        return arr.to_numpy()  # assume a 1D array
    else:
        dim1 = len(arr)  # assume a 2D array
        return np.asarray(arr.values.to_numpy()).reshape(dim1, -1)


def arrow_batch_to_table(batch_or_tbl: Union[pa.Table, pa.RecordBatch]) -> pa.Table:
    if not isinstance(batch_or_tbl, pa.Table):
        return pa.Table.from_batches([batch_or_tbl])
    else:
        return batch_or_tbl


def table_arrow_to_np_arrays(tbl: pa.Table, cols: list[str]) -> list[np.ndarray]:
    return [array_arrow_to_np(tbl[c]) for c in cols]


def np_arrays_to_arrow_table(arrays: list[np.ndarray], cols: list[str]) -> pa.Table:
    return pa.Table.from_pydict(
        {c: array_np_to_arrow(arr) for c, arr in zip(cols, arrays)}
    )


# -----------------------------------------------------------
# base classes for operators
# -----------------------------------------------------------


@runtime_checkable
class Operator(Protocol):
    """Base class for all operators."""

    def make_actor(self) -> "OperatorExecutor":
        """Create a Ray actor for this operator."""
        raise NotImplementedError


@runtime_checkable
class OperatorExecutor(Protocol):
    """Base class for all operator executors."""

    op: Operator

    def run(self, *args, **kwargs) -> pa.Table:
        raise NotImplementedError
