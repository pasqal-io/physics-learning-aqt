from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTEstimator


# TODO convert to a pydantic BaseModel and include validation
class GradientPSR:
    def __init__(
        self,
        # TODO make backend name an Enum
        backend_name: str = "offline_simulator_no_noise",
        circuit_transpile_level: int = 3,
    ) -> None:
        """Sets the backend and estimator with the right transpile level for the circuit."""
        self._backend = AQTProvider().get_backend(backend_name)

        self._estimator = AQTEstimator(self._backend)
        self._estimator.set_transpile_options(optimization_level=circuit_transpile_level)
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
