from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from physicslearningaqt.ansatz import HEA
from physicslearningaqt.gradient import GradientPSR


def main() -> None:
    rng = np.random.default_rng(seed=42)
    n_qubits = 3
    n_layers = 2

    g_psr = GradientPSR()

    hea = HEA(n_qubits, n_layers)

    H = SparsePauliOp("Z" * 3)

    grad_values = g_psr.run(hea.circuit, H, hea.get_random_params(rng))
    print("State estimator gradient computed with parameter shift", grad_values)


if __name__ == "__main__":
    main()
