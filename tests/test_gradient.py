from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from physicslearningaqt.gradient import BackendName, GradientPSR


@pytest.mark.parametrize("transpile_level", [-1, 4])
def test_gradientpsr_wrong_transpile_level(transpile_level):
    with pytest.raises(ValidationError):
        GradientPSR(
            backend_name=BackendName.OFFLINE_SIM_NO_NOISE, circuit_transpile_level=transpile_level
        )


@pytest.mark.parametrize(
    "backend_name", [BackendName.OFFLINE_SIM_NO_NOISE, BackendName.OFFLINE_SIM_NOISE]
)
def test_gradientpsr_run(
    mock_parametrized_quantum_circuit, mock_observable, mock_parameters, backend_name
):
    gpsr = GradientPSR(backend_name=backend_name)
    gradient = gpsr.run(mock_parametrized_quantum_circuit, mock_observable, mock_parameters)[0]

    assert isinstance(gradient, np.ndarray)
    assert gradient.size == mock_parameters.size
    assert (gradient >= -1.0).all()
    assert (gradient <= 1.0).all()
