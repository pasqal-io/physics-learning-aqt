from __future__ import annotations

from functools import partial
from typing import Callable, Protocol

import numpy as np


def generate_1d_data(
    function: Callable[[np.ndarray], np.ndarray],
    size: int,
    interval: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = interval
    x = rng.random(size) * (x_max - x_min) + x_min
    return x, function(x)


class DataGenerator(Protocol):
    def __call__(self, size: int, *, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        ...


generate_sine = partial(generate_1d_data, function=np.sin, interval=(-np.pi, np.pi))
