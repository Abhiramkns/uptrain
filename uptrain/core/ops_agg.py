"""Operators to compute metrics over embeddings"""

from typing import Any, Callable, Literal, Union
from itertools import product
from functools import cache

import duckdb
import numba
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel
import ray

from .ops_base import (
    Operator,
    OperatorExecutor,
    ReduceOp,
    arrow_batch_to_table,
    np_arrays_to_arrow_table,
    table_arrow_to_np_arrays,
)

# -----------------------------------------------------------
# Window Aggregation Operators (N-to-N)
# -----------------------------------------------------------


class WindowAggOp(BaseModel):
    """Operator to compute a window aggregation over a column in a table (N-to-N)."""

    id_col: str  # aggregations are done per id
    value_col: str  # the column to aggregate over
    seq_col: str  # the column to sort by before aggregation


# @ray.remote
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
        self.compute_loop = generate_compute_loop_for_window_agg(compute_op)

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


def compute_op_l2dist_initial(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    return np.sum(np.square(value - interm_value)), interm_value  # type: ignore


def compute_op_l2dist_running(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    return np.sum(np.square(value - interm_value)), value  # type: ignore


def compute_op_cosine_dist_initial(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    cos_similarity = np.dot(value, interm_value) / (
        np.linalg.norm(value) * np.linalg.norm(interm_value)
    )
    return (1 - cos_similarity), interm_value


def compute_op_cosine_dist_running(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    cos_similarity = np.dot(value, interm_value) / (
        np.linalg.norm(value) * np.linalg.norm(interm_value)
    )
    return (1 - cos_similarity), value


@cache
def generate_compute_loop_for_window_agg(compute_fn):
    """Generates a compute loop for the given function."""

    inlined_fn = numba.njit(inline="always")(compute_fn)

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

            output_value_array[idx], interm_value = inlined_fn(_value, interm_value)  # type: ignore
        new_interm_value_array[interm_idx] = interm_value

        return output_value_array, new_interm_value_array

    return compute_loop


class L2Dist(WindowAggOp):
    """Computes the L2 distance between the current embedding and the initial/previous one."""

    mode: Literal["initial", "running"]

    def make_actor(self):
        mode = str(
            self.mode
        )  # appease pylance since it doesn't let me compare a Literal to a string
        if mode == "initial":
            return WindowAggExecutor(self, compute_op_l2dist_initial)
        elif mode == "running":
            return WindowAggExecutor(self, compute_op_l2dist_running)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class CosineDist(WindowAggOp):
    """Computes the cosine distance between the current embedding and the initial/previous one."""

    mode: Literal["initial", "running"]

    def make_actor(self):
        mode = str(self.mode)
        if mode == "initial":
            return WindowAggExecutor(self, compute_op_cosine_dist_initial)
        elif mode == "running":
            return WindowAggExecutor(self, compute_op_cosine_dist_running)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# -----------------------------------------------------------
# Pairwise metrics computed over the full table at once (N-to-1)
# -----------------------------------------------------------


class CosineDistHistogram(ReduceOp):
    """Computes the histogram of cosine distances between all embeddings in the table."""

    def make_actor(self):
        return PairwiseMetricExecutor.remote(self, compute_op_cosine_dist_initial)


@ray.remote
class PairwiseMetricExecutor:
    """Executor for computing a metric between pairs of embeddings."""

    op: ReduceOp
    compute_func: Callable

    def __init__(self, op: ReduceOp, compute_op: Callable) -> None:
        self.op = op
        self.compute_fn = compute_op


# -----------------------------------------------------------
# Partition Operator
# -----------------------------------------------------------


class PartitionOp(BaseModel, arbitrary_types_allowed=True):
    """Operator to partition a table/batch based on values in one/multiple columns,
    compute an aggregate on each partition, and combine the results.
    """

    col_values: dict[str, list[Any]]  # mapping of columns to corresponding values
    agg_op: Operator  # operator to apply to each partition

    def make_actor(self):
        return PartitionExecutor.remote(self)


@ray.remote
class PartitionExecutor:
    """Executor for partitioning."""

    op: PartitionOp
    list_cols: list[str]
    list_value_tuples: list[tuple[Any, ...]]
    list_agg_actors: list[OperatorExecutor]

    def __init__(self, op: PartitionOp) -> None:
        self.op = op
        self.list_cols = list(op.col_values.keys())
        self.list_value_tuples = list(product(*op.col_values.values()))
        self.list_agg_actors = [
            self.op.agg_op.make_actor() for _ in self.list_value_tuples
        ]

    def run(self, batch: Union[pa.Table, pa.RecordBatch]) -> pa.Table:
        if batch is None:
            return None

        tbl = arrow_batch_to_table(batch)
        result_refs = []
        for idx, values in enumerate(self.list_value_tuples):
            filter_expr = None
            for col, value in zip(self.list_cols, values):
                expr = pc.field(col) == value
                if filter_expr is None:
                    filter_expr = expr
                else:
                    filter_expr = filter_expr & expr

            agg_actor = self.list_agg_actors[idx]
            result_refs.append(agg_actor.run(tbl.filter(filter_expr)))

        # TODO: for too many col-value tuples, buffer ray results and concat in batches
        # link - https://docs.ray.io/en/latest/ray-core/patterns/ray-get-too-many-objects.html

        # working with remote actors or nay?
        # all_tables = [tbl for tbl in ray.get(result_refs) if tbl is not None]
        all_tables = [tbl for tbl in result_refs if tbl is not None]
        return pa.concat_tables(all_tables)
