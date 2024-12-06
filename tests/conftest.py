from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp

_num_qubits = 3
_num_ansatz_reps = 2


@pytest.fixture
def mock_parametrized_quantum_circuit():
    qc = EfficientSU2(
        num_qubits=_num_qubits,
        su2_gates=["rx"],
        reps=_num_ansatz_reps,
        skip_final_rotation_layer=True,
    )
    return qc


@pytest.fixture
def mock_observable():
    return SparsePauliOp("XZZ")


@pytest.fixture
def mock_parameters():
    rng = np.random.default_rng(0)
    return rng.random((1, _num_qubits * _num_ansatz_reps)) * 2 * np.pi - np.pi
