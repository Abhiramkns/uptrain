"""Operators for the uptrain.core module. 

- Arrow/Numpy conversion utils - since we leverage duckdb as a cache and ray for execution, intermediate 
outputs are stored as Arrow batches. 
"""

from typing import Union, Callable, Any, Protocol, runtime_checkable

import duckdb
import numpy as np
import numba
import pyarrow as pa
from pydantic import BaseModel
import ray

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


class WindowAggOp(BaseModel):
    """Operator to compute a window aggregation over a column in a table (N-to-N)."""

    id_col: str  # aggregations are done per id
    value_col: str  # the column to aggregate over
    seq_col: str  # the column to sort by before aggregation


@ray.remote
class WindowAggExecutor(OperatorExecutor):
    """Executor for window aggregations."""

    op: WindowAggOp
    cache_conn: duckdb.DuckDBPyConnection
    compute_loop: Callable

    def __init__(self, op: WindowAggOp, compute_op: Callable) -> None:
        """Initialize a Ray actor for the given operator.

        Args:
            op: Operator to execute.
            compute_op: Function to computes the aggregate value for the given value and intermediate value.
                The executor batches calls to this function and jit compiles the loop using numba.
        """
        self.op = op
        self.cache_conn = duckdb.connect(":memory:")
        self.cache_conn.execute(
            f"CREATE TABLE intermediates (id LONG PRIMARY KEY, value FLOAT[]);"
        )  # TODO: support other types

        self.compute_loop = self._generate_compute_loop(
            numba.njit(inline="always")(compute_op)
        )

    @classmethod
    def _generate_compute_loop(cls, compute_fn):
        """Generates a compute loop for the given function."""

        @numba.njit
        def compute_loop(id_array, value_array, interm_id_array, interm_value_array):
            output_value_array = np.zeros(len(id_array))
            new_interm_value_array = np.zeros_like(interm_value_array)

            interm_idx = 0
            interm_value = interm_value_array[interm_idx]
            for idx, (_id, _value) in enumerate(zip(id_array, value_array)):
                if _id != interm_id_array[interm_idx]:
                    new_interm_value_array[interm_idx] = interm_value
                    interm_idx += 1
                    interm_value = interm_value_array[interm_idx]

                output_value_array[idx], interm_value = compute_fn(_value, interm_value)
            new_interm_value_array[interm_idx] = interm_value

            return output_value_array, new_interm_value_array

        return compute_loop

    def fetch_interm_state(
        self, batch: Union[pa.Table, pa.RecordBatch]
    ) -> list[np.ndarray]:
        tbl = arrow_batch_to_table(batch)
        self.cache_conn.execute(
            """
            SELECT * FROM (
                SELECT id, value
                FROM intermediates
                WHERE id IN (SELECT {id_col} from tbl)
                
                UNION
            
                SELECT {id_col}, FIRST({value_col} ORDER BY {sort_col} ASC)
                FROM tbl 
                WHERE {id_col} NOT IN (SELECT id FROM intermediates)
                GROUP BY {id_col}
            )
            ORDER BY 1 ASC;
            """.format(
                id_col=self.op.id_col,
                value_col=self.op.value_col,
                sort_col=self.op.seq_col,
            )
        )
        interm_state = self.cache_conn.fetch_arrow_table()
        return table_arrow_to_np_arrays(interm_state, ["id", "value"])

    def update_interm_state(self, id_array: np.ndarray, value_array: np.ndarray):
        tbl = np_arrays_to_arrow_table([id_array, value_array], ["id", "value"])

        # upserts don't work for List data types, so we gotta do it in steps.
        self.cache_conn.execute(
            "DELETE FROM intermediates WHERE id IN (SELECT id FROM tbl);"
        )
        self.cache_conn.execute("INSERT INTO intermediates SELECT id, value FROM tbl;")

    def run(self, tbl: Union[pa.Table, pa.RecordBatch]) -> pa.Table:
        tbl = arrow_batch_to_table(tbl)
        sorted_tbl = tbl.sort_by(
            [(self.op.id_col, "ascending"), (self.op.seq_col, "ascending")]
        )

        id_array, value_array = table_arrow_to_np_arrays(
            sorted_tbl, [self.op.id_col, self.op.value_col]
        )
        interm_id_array, interm_value_array = self.fetch_interm_state(sorted_tbl)

        output_value_array, new_interm_value_array = self.compute_loop(
            id_array, value_array, interm_id_array, interm_value_array
        )
        self.update_interm_state(interm_id_array, new_interm_value_array)
        return sorted_tbl.append_column("aggregate", pa.array(output_value_array))


class ReduceOp(BaseModel):
    """Operators that produce a single row of output (N-to-1)."""

    value_col: str  # the column to aggregate over
