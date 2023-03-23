"""Operators to compute metrics over embeddings"""

from typing import Optional

import duckdb
import os
import pyarrow as pa
from pydantic import BaseModel
import ray

from .ops_base import Operator, OperatorExecutor, arrow_batch_to_table


class ParquetWriter(BaseModel, Operator):
    dir_path: str  # appending to a parquet file isn't supported using pyarrow

    def make_actor(self):
        return ParquetWriterExecutor.remote(self)


@ray.remote
class ParquetWriterExecutor(OperatorExecutor):
    op: ParquetWriter
    count: int  # number of batches written

    def __init__(self, op: ParquetWriter):
        self.op = op
        self.count = 0
        os.makedirs(os.path.dirname(self.op.dir_path), exist_ok=True)

    def run(self, tbl: pa.Table) -> pa.Table:
        self.count += 1
        pa.parquet.write_table(
            tbl, os.path.join(self.op.dir_path, f"{self.count}.parquet")
        )


class DuckdbReader(BaseModel, Operator):
    fpath: str
    query: str

    def make_actor(self):
        return DuckdbReaderExecutor.remote(self)


@ray.remote
class DuckdbReaderExecutor(OperatorExecutor):
    op: DuckdbReader
    conn: duckdb.DuckDBPyConnection
    reader: pa.RecordBatchReader

    def __init__(self, op: DuckdbReader):
        self.op = op
        self.conn = duckdb.connect(database=self.op.fpath, read_only=True)
        self.conn.execute(self.op.query)
        self.reader = self.conn.fetch_record_batch()

    def run(self) -> Optional[pa.Table]:
        try:
            return arrow_batch_to_table(self.reader.read_next_batch())
        except StopIteration:
            return None
