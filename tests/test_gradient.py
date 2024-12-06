from __future__ import annotations

import pytest
from pydantic import ValidationError

from physicslearningaqt.gradient import BackendName, GradientPSR


@pytest.mark.parametrize("transpile_level", [-1, 4])
def test_gradientpsr_wrong_transpile_level(transpile_level):
    with pytest.raises(ValidationError):
        GradientPSR(
            backend_name=BackendName.OFFLINE_SIM_NO_NOISE, circuit_transpile_level=transpile_level
        )
