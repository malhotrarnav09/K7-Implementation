"""Print the complete K7 circuit directly in the terminal."""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from src.K7.bases import bases


ALICE_INPUT = 2
BOB_INPUT = 3


def embed_in_eight_dimensions(matrix):
    embedded = np.identity(8, dtype=complex)
    embedded[:6, :6] = matrix
    return embedded


bases_eight = [
    embed_in_eight_dimensions(matrix)
    for matrix in bases
]


def build_circuit(alice_input, bob_input):
    circuit = QuantumCircuit(6)

    # Prepare the six-state superposition on Alice's three qubits.
    preparation = QuantumCircuit(
        3,
        name="Phi6_seed",
    )

    preparation.initialize(
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 2],
        normalize=True,
    )

    # Display the exact symbolic state instead of decimal amplitudes.
    preparation_instruction = preparation.to_instruction(
        label="1/sqrt(6)(|0>+|1>+|2>+|3>+|4>+|5>)"
    )

    circuit.append(
        preparation_instruction,
        [0, 1, 2],
    )

    # Entangle Alice's and Bob's three-qubit registers.
    circuit.cx(0, 3)
    circuit.cx(1, 4)
    circuit.cx(2, 5)

    # Construct the two embedded 8x8 measurement matrices.
    alice_matrix = bases_eight[alice_input - 1]

    bob_matrix = np.conjugate(
        bases_eight[bob_input - 1]
    )

    # Use simple labels in the terminal diagram.
    unitary_1 = UnitaryGate(
        alice_matrix,
        label="Unitary 1",
    )

    unitary_2 = UnitaryGate(
        bob_matrix,
        label="Unitary 2",
    )

    circuit.append(
        unitary_1,
        [0, 1, 2],
    )

    circuit.append(
        unitary_2,
        [3, 4, 5],
    )

    circuit.measure_all()

    return circuit


if __name__ == "__main__":
    circuit = build_circuit(
        ALICE_INPUT,
        BOB_INPUT,
    )

    print("\nCOMPLETE SIX-QUBIT K7 CIRCUIT\n")

    print(
        circuit.draw(
            output="text",
            fold=180,
        )
    )

    print(
        "\nGate counts:",
        dict(circuit.count_ops()),
    )