from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.gradients import ParamShiftEstimatorGradient
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.primitives import AQTEstimator


def main() -> None:
    rng = np.random.default_rng(seed=42)

    backend_name = "offline_simulator_no_noise"

    provider = AQTProvider()
    backend = provider.get_backend(backend_name)

    estimator = AQTEstimator(backend)
    estimator.set_transpile_options(optimization_level=3)  # embedded transpilation

    gradient_estimator = ParamShiftEstimatorGradient(estimator)

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

    grad_values = gradient_estimator.run(qc, H, params_values).result().gradients
    print("State estimator gradient computed with parameter shift", grad_values)


if __name__ == "__main__":
    main()
