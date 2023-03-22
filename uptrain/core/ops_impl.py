"""Operators to compute metrics over embeddings"""

import numpy as np
import pyarrow as pa

from .ops_base import AggregateOperator, StatelessOperator


class Sorter(StatelessOperator):
    """Sorts an arrow table by the given columns."""

    def __init__(self, cols_n_order: list[tuple[str, bool]]):
        self.cols_n_order = cols_n_order

    def run(self, tbl: pa.Table) -> pa.Table:
        return tbl.sort_by(
            [
                (col, "ascending" if ascending else "descending")
                for col, ascending in self.cols_n_order
            ]
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cols_n_order})"


class L2Dist_Initial(AggregateOperator):
    """Computes the L2 distance between the current embedding and the initial one."""

    @staticmethod
    def compute_op(
        value: np.ndarray, interm_value: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return np.sum(np.square(value - interm_value)), interm_value


class L2Dist_Running(AggregateOperator):
    """Computes the L2 distance between the current embedding and the previous one."""

    @staticmethod
    def compute_op(
        value: np.ndarray, interm_value: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return np.sum(np.square(value - interm_value)), value


class CosineDist_Running(AggregateOperator):
    """Computes the cosine distance between the current embedding and the previous one."""

    @staticmethod
    def compute_op(
        value: np.ndarray, interm_value: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return (
            1
            - np.dot(value, interm_value)
            / (np.linalg.norm(value) * np.linalg.norm(interm_value)),
            value,
        )


class CosineDist_Initial(AggregateOperator):
    """Computes the cosine distance between the current embedding and the initial one."""

    @staticmethod
    def compute_op(
        value: np.ndarray, interm_value: np.ndarray
    ) -> tuple[float, np.ndarray]:
        return (
            1
            - np.dot(value, interm_value)
            / (np.linalg.norm(value) * np.linalg.norm(interm_value)),
            interm_value,
        )
