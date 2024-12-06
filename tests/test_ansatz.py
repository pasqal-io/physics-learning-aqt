from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit

from physicslearningaqt.ansatz import HEA


def test_hea():
    rng = np.random.default_rng(42)
    hea = HEA(n_qubits=3, n_layers=2)
    assert isinstance(hea.circuit, QuantumCircuit)
    assert hea.circuit.num_parameters == 6

    rp = hea.get_random_params(rng)
    assert rp.size == hea.circuit.num_parameters
