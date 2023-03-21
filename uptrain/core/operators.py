"""Operators for the uptrain.core module. 

- Arrow/Numpy conversion utils - since we leverage duckdb as a cache and ray for execution, intermediate 
outputs are stored as Arrow batches. 
"""

import duckdb
import numpy as np
import pyarrow as pa
from typing import Union


def array_np_to_arrow(arr: np.ndarray) -> pa.Array:
    assert arr.ndim in (1, 2), "Only 1D and 2D arrays are supported."
    if arr.ndim == 1:
        return pa.array(arr)
    else:
        dim1, dim2 = arr.shape
        return pa.ListArray.from_arrays(
            np.arange(0, (dim1 + 1) * dim2, dim2), arr.ravel()
        )


def array_arrow_to_np(arr: pa.ListArray) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if not pa.types.is_list(arr.type):
        return arr.to_numpy(zero_copy_only=True)  # assume a 1D array
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


class StatelessOperator:
    """Base class for stateless operators."""

    def run(self, *args, **kwargs) -> pa.Table:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SortTable(StatelessOperator):
    """Sorts a table by a given column."""

    def __init__(self, col: str, ascending: bool = True):
        self.col = col
        self.ascending = ascending

    def run(self, tbl: pa.Table) -> pa.Table:
        return tbl.sort_by(
            [(self.col, "ascending" if self.ascending else "descending")]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(col={self.col}, ascending={self.ascending})"


class L2Dist_Running:
    """Computes the L2 distance for the vector value at each id wrt to the previous value."""

    id_col: str
    value_col: str
    cache_conn: duckdb.DuckDBPyConnection

    def __init__(self, id_col: str, value_col: str, sort_col: str) -> None:
        self.id_col = id_col
        self.value_col = value_col
        self.sort_col = sort_col

        self.cache_conn = duckdb.connect(":memory:")
        self.cache_conn.execute(
            f"CREATE TABLE intermediates (id LONG PRIMARY KEY, value FLOAT[]);"
        )  # TODO: support other types

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
            ORDER BY 1
            """.format(
                id_col=self.id_col,
                value_col=self.value_col,
                sort_col=self.sort_col,
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
        """We assume the input table is sorted by id, through usage of a Sort operator before."""
        tbl = arrow_batch_to_table(tbl)
        id_array, value_array = table_arrow_to_np_arrays(
            tbl, [self.id_col, self.value_col]
        )
        interm_id_array, interm_value_array = self.fetch_interm_state(tbl)

        output_value_array = np.zeros(len(id_array))
        new_interm_value_array = np.zeros_like(interm_value_array)

        interm_idx = 0
        interm_value = interm_value_array[interm_idx]
        for idx, (_id, _value) in enumerate(zip(id_array, value_array)):
            if _id != interm_id_array[interm_idx]:
                new_interm_value_array[interm_idx] = interm_value
                interm_idx += 1
                interm_value = interm_value_array[interm_idx]

            output_value_array[idx] = np.sum(np.square(_value - interm_value))
        new_interm_value_array[interm_idx] = interm_value

        self.update_interm_state(interm_id_array, new_interm_value_array)
        return np_arrays_to_arrow_table([id_array, output_value_array], ["id", "value"])
