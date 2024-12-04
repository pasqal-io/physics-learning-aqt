from __future__ import annotations

import numpy as np
from qiskit.circuit import ParameterVector, QuantumCircuit, QuantumRegister


class HEA:
    """Hardware Efficient Ansatz."""

    def __init__(self, n_qubits: int, n_layers: int) -> None:
        self._reg = QuantumRegister(n_qubits)
        self._circ = QuantumCircuit(self._reg)
        self._params = ParameterVector("theta", n_qubits * n_layers)

        for ll in range(n_layers):
            for qq in range(n_qubits):
                self._circ.rx(self._params[qq + ll * n_qubits], qq)
            for q0, q1 in zip(range(n_qubits - 1), range(1, n_qubits)):
                self._circ.cz(q0, q1)

        self._circ.measure_all()

    @property
    def circuit(self) -> QuantumCircuit:
        return self._circ

    def get_random_params(self, rng: np.random.Generator) -> np.ndarray:
        return rng.random((1, len(self._params))) * 2 * np.pi - np.pi
