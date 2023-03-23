"""Operators to compute metrics over embeddings"""

from typing import Literal

import numpy as np
import pyarrow as pa
from pydantic import BaseModel

from .ops_base import WindowAggOp, WindowAggExecutor


def compute_op_l2dist_initial(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    return np.sum(np.square(value - interm_value)), interm_value


def compute_op_l2dist_running(
    value: np.ndarray, interm_value: np.ndarray
) -> tuple[float, np.ndarray]:
    return np.sum(np.square(value - interm_value)), value


class L2Dist(WindowAggOp):
    """Computes the L2 distance between the current embedding and the initial/previous one."""

    mode: Literal["initial", "running"]

    def make_actor(self):
        mode = str(
            self.mode
        )  # appease pylance since it doesn't let me compare a Literal to a string
        if mode == "initial":
            return WindowAggExecutor.remote(self, compute_op_l2dist_initial)
        elif mode == "running":
            return WindowAggExecutor.remote(self, compute_op_l2dist_running)
        else:
            raise ValueError(f"Unknown mode: {mode}")


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


class CosineDist(WindowAggOp):
    """Computes the cosine distance between the current embedding and the initial/previous one."""

    mode: Literal["initial", "running"]

    def make_actor(self):
        mode = str(self.mode)
        if mode == "initial":
            return WindowAggExecutor.remote(self, compute_op_cosine_dist_initial)
        elif mode == "running":
            return WindowAggExecutor.remote(self, compute_op_cosine_dist_running)
        else:
            raise ValueError(f"Unknown mode: {mode}")
