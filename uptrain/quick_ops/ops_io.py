"""Operators to compute metrics over embeddings"""

from typing import Optional

import duckdb
import os
import pyarrow as pa
from pydantic import BaseModel
import ray

from .ops_base import arrow_batch_to_table


class ParquetWriter(BaseModel):
    dir_path: str  # appending to a parquet file isn't supported using pyarrow

    def make_actor(self):
        return ParquetWriterExecutor.remote(self)


@ray.remote
class ParquetWriterExecutor:
    op: ParquetWriter
    count: int  # number of batches written

    def __init__(self, op: ParquetWriter):
        self.op = op
        self.count = 0
        os.makedirs(self.op.dir_path, exist_ok=True)

    def run(self, tbl: pa.Table) -> None:
        import pyarrow.parquet

        self.count += 1
        pa.parquet.write_table(
            tbl, os.path.join(self.op.dir_path, f"{self.count}.parquet")
        )


class DuckdbReader(BaseModel):
    fpath: str
    query: str

    def make_actor(self):
        return DuckdbReaderExecutor.remote(self)


@ray.remote
class DuckdbReaderExecutor:
    op: DuckdbReader
    conn: duckdb.DuckDBPyConnection
    reader: pa.RecordBatchReader

    def __init__(self, op: DuckdbReader):
        self.op = op
        self.conn = duckdb.connect(database=self.op.fpath)
        self.conn.execute(self.op.query)
        self.reader = self.conn.fetch_record_batch()

    def run(self) -> Optional[pa.Table]:
        try:
            return arrow_batch_to_table(self.reader.read_next_batch())
        except StopIteration:
            return None


class CsvReader(BaseModel):
    fpath: str
    columns: Optional[list[str]] = None

    def make_actor(self):
        return CsvReaderExecutor.remote(self)


@ray.remote
class CsvReaderExecutor:
    op: CsvReader
    dataset: pa.Table
    rows_read: int
    batch_size: int = 50_000

    def __init__(self, op: CsvReader):
        import pyarrow.csv

        self.op = op
        self.dataset = pa.csv.read_csv(self.op.fpath)
        self.rows_read = 0
        self.batch_size = 50_000

    def run(self) -> Optional[pa.Table]:
        if self.rows_read >= len(self.dataset):
            return None
        else:
            self.rows_read += self.batch_size
            return self.dataset.slice(self.rows_read, self.batch_size)
