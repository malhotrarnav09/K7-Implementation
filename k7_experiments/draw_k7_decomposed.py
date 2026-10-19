"""Display decomposed K7 unitaries without rotation-angle labels."""

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

from k7_experiments.draw_k7_compact import (
    ALICE_INPUT,
    BOB_INPUT,
    bases_eight,
    build_circuit,
)


# Decompose into rotation gates and CX gates.
NATIVE_GATES = ["rz", "ry", "rx", "cx"]


def decompose_unitary(matrix):
    """Create and verify the real numerical decomposition."""

    unitary_circuit = QuantumCircuit(3)

    unitary_circuit.append(
        UnitaryGate(matrix),
        [0, 1, 2]
    )

    decomposed = transpile(
        unitary_circuit,
        basis_gates=NATIVE_GATES,
        optimization_level=3,
        seed_transpiler=8,
    )

    # Verify the real circuit before hiding its angles.
    if not Operator(decomposed).equiv(Operator(matrix)):
        raise RuntimeError(
            "Transpilation changed the unitary"
        )

    return decomposed


def hide_rotation_angles(circuit):
    """
    Create a display-only copy with the rotation angles removed.

    The original numerical circuit remains unchanged.
    """

    display = QuantumCircuit(circuit.num_qubits)

    for instruction in circuit.data:
        operation = instruction.operation

        qubit_indices = [
            circuit.find_bit(qubit).index
            for qubit in instruction.qubits
        ]

        display_qubits = [
            display.qubits[index]
            for index in qubit_indices
        ]

        if operation.name == "cx":
            display.cx(
                display_qubits[0],
                display_qubits[1]
            )

        elif operation.name in {"rx", "ry", "rz"}:
            angle_free_gate = Gate(
                operation.name,
                1,
                [],
                label=operation.name.upper(),
            )

            display.append(
                angle_free_gate,
                [display_qubits[0]]
            )

        else:
            angle_free_gate = Gate(
                operation.name,
                len(display_qubits),
                [],
                label=operation.name.upper(),
            )

            display.append(
                angle_free_gate,
                display_qubits
            )

    return display


def main():
    # Select the matrices corresponding to Alice and Bob's vertices.
    alice_matrix = bases_eight[ALICE_INPUT - 1]

    bob_matrix = np.conjugate(
        bases_eight[BOB_INPUT - 1]
    )

    # These circuits contain the real numerical rotation angles.
    alice_decomposed = decompose_unitary(
        alice_matrix
    )

    bob_decomposed = decompose_unitary(
        bob_matrix
    )

    # These copies are only used for the readable diagrams.
    alice_display = hide_rotation_angles(
        alice_decomposed
    )

    bob_display = hide_rotation_angles(
        bob_decomposed
    )

    print(
        f"\nALICE U_{ALICE_INPUT}: DECOMPOSED GATE STRUCTURE "
        "(ANGLES OMITTED)\n"
    )

    print(
        alice_display.draw(
            output="text",
            fold=160
        )
    )

    print(
        "\nAlice real gate counts:",
        dict(alice_decomposed.count_ops())
    )

    print(
        "Alice real depth:",
        alice_decomposed.depth()
    )

    input(
        "\nPress Enter to display Bob's gate structure..."
    )

    print(
        f"\nBOB conj(U_{BOB_INPUT}): DECOMPOSED GATE STRUCTURE "
        "(ANGLES OMITTED)\n"
    )

    print(
        bob_display.draw(
            output="text",
            fold=160
        )
    )

    print(
        "\nBob real gate counts:",
        dict(bob_decomposed.count_ops())
    )

    print(
        "Bob real depth:",
        bob_decomposed.depth()
    )

    # Calculate resource numbers from the real complete circuit.
    complete_circuit = build_circuit(
        ALICE_INPUT,
        BOB_INPUT
    )

    complete_transpiled = transpile(
        complete_circuit,
        basis_gates=NATIVE_GATES,
        optimization_level=3,
        seed_transpiler=8,
    )

    print(
        "\nCOMPLETE SIX-QUBIT RESOURCE SUMMARY\n"
    )

    print(
        "Gate counts:",
        dict(complete_transpiled.count_ops())
    )

    print(
        "Circuit depth:",
        complete_transpiled.depth()
    )

    print(
        "\nNOTE: Rotation angles were omitted only "
        "from the displayed diagrams."
    )

    print(
        "The verified numerical angles remain "
        "in the real transpiled circuits."
    )


if __name__ == "__main__":
    main()
