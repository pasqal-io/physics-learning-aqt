from __future__ import annotations

from enum import Enum, auto
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTEstimator


class BackendName(Enum):
    OFFLINE_SIM_NO_NOISE = auto()
    OFFLINE_SIM_NOISE = auto()


_backend_name_mapping = {
    BackendName.OFFLINE_SIM_NO_NOISE: "offline_simulator_no_noise",
    BackendName.OFFLINE_SIM_NOISE: "offline_simulator_noise",
}


class GradientPSR(BaseModel):
    backend_name: BackendName
    circuit_transpile_level: int = Field(default=3, ge=0, le=3, strict=True)

    def model_post_init(self, __context: Any) -> None:
        """Sets the backend and estimator with the right transpile level for the circuit."""
        self._backend = AQTProvider().get_backend(_backend_name_mapping[self.backend_name])

        self._estimator = AQTEstimator(self._backend)
        self._estimator.set_transpile_options(optimization_level=self.circuit_transpile_level)
        self._gradient_estimator = ParamShiftEstimatorGradient(self._estimator)

    def run(
        self,
        quantum_circuit: QuantumCircuit,
        observable: SparsePauliOp,
        parameter_values: np.ndarray,
    ) -> np.ndarray:
        return (
            self._gradient_estimator.run(quantum_circuit, observable, parameter_values)
            .result()
            .gradients
        )
