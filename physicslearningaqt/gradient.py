from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
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


def main() -> None:
    rng = np.random.default_rng(seed=42)

    g_psr = GradientPSR()

    def hea(n_qubits: int, n_layers: int) -> QuantumCircuit:
        params = ParameterVector("theta", n_qubits * n_layers)

        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr)

        for ll in range(n_layers):
            for qq in range(n_qubits):
                qc.rx(params[qq + ll * n_qubits], qq)
            for q0, q1 in zip(range(n_qubits - 1), range(1, n_qubits)):
                qc.cz(q0, q1)

        qc.measure_all()

        return qc

    qc = hea(3, 2)

    H = SparsePauliOp("Z" * 3)

    params_values = rng.random((1, 6)) * 2 * np.pi

    grad_values = g_psr.run(qc, H, params_values)
    print("State estimator gradient computed with parameter shift", grad_values)


if __name__ == "__main__":
    main()
