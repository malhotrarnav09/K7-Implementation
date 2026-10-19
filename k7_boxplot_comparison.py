from qiskit import transpile

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_aer.noise import ReadoutError

from qiskit_aer.noise import pauli_error

from src.K7.verifier import vertex_edges, game_result

import matplotlib.pyplot as plt

import csv

from quantem.utils import convert_to_PCS_circ_largest_clifford
from src.K7.quantumcircuit import build_game_circuit

from qiskit import ClassicalRegister

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from quantem import QEDCompiler, QEDStrategy

P=0.001

noise_model = NoiseModel()

cx_error = depolarizing_error(16 * P / 15, 2)

noise_model.add_all_qubit_quantum_error(
    cx_error, ["cx"]
)

#noise for the single qubit gates, rx, ry, rz
single_error = depolarizing_error(
    4 * (P / 10) / 3, 1
    )

noise_model.add_all_qubit_quantum_error(
    single_error,
    ["rx", "ry", "rz"]
    )

# incoporating redout issues as another form of error
measurement_error = ReadoutError([
    [1 - 5 * P, 5 * P],
    [5 * P, 1 - 5 * P]
])

noise_model.add_all_qubit_readout_error(
    measurement_error
)

#adding pauli error to the noise model eg. initialization error, reset error, etc.
prep_error = pauli_error([
    ("X", 2 * P),
    ("I", 1 - 2 * P)
])

noise_model.add_all_qubit_quantum_error(
    prep_error,
    ["reset"]
)

from draw_k7_compact import (
    ALICE_INPUT,
    BOB_INPUT,
    build_circuit,
)

from draw_k7_decomposed import NATIVE_GATES

complete_circuit = build_circuit(
    ALICE_INPUT,
    BOB_INPUT,
)

complete_transpiled = transpile(
    complete_circuit,
    basis_gates=NATIVE_GATES,
    optimization_level=3,
    seed_transpiler=8,
)

print("Gate counts:", dict(complete_transpiled.count_ops()))
print("Circuit depth:", complete_transpiled.depth())

print(
    "Reset instructions:",
    complete_transpiled.count_ops().get("reset", 0)
)

input_pairs = []

for a in range(1, 8):
    for b in range(1, 8):
        input_pairs.append((a, b))

simulator = AerSimulator(noise_model=noise_model)

win_rates = []

for a, b in input_pairs:

    complete_circuit = build_circuit(a, b)

    complete_transpiled = transpile(
        complete_circuit,
        basis_gates=NATIVE_GATES,
        optimization_level=3,
        seed_transpiler=8,
    )

    job = simulator.run(
        complete_transpiled,
        shots=1024,
        seed_simulator = 42
    )

    counts = job.result().get_counts()

    winning_shots = 0

    for bitstring, frequency in counts.items():

        bob_index = int(bitstring[:3], 2)
        alice_index = int(bitstring[3:], 2)

        if alice_index >= 6 or bob_index >= 6:
            continue

        alice_edge = vertex_edges[a][alice_index]
        bob_edge = vertex_edges[b][bob_index]

        if game_result(alice_edge, bob_edge) == 1:
            winning_shots += frequency

    win_probability = winning_shots / sum(counts.values())

    win_rates.append(win_probability)

    print(f"({a}, {b}): {win_probability:.2%}")

#csv file to store the results
with open("k7_baseline_results.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["Alice", "Bob", "Win Probability"])

    for (a, b), rate in zip(input_pairs, win_rates):
        writer.writerow([a, b, rate])

#quantem implementation
NUM_DATA_QUBITS = 6
NUM_CHECKS = 1

iceberg_circuits = {}

for a, b in input_pairs:

    original = build_game_circuit(a, b)

    # Rebuild K7 using unitary state preparation
    payload = QuantumCircuit(6)

    payload.append(
        StatePreparation(
            [1, 1, 1, 1, 1, 1, 0, 0],
            normalize=True
        ),
        [0, 1, 2]
    )

    for instruction in original.data:

        if instruction.operation.name == "initialize":
            continue

        indices = [
            original.find_bit(q).index
            for q in instruction.qubits
        ]

        payload.append(
            instruction.operation,
            indices
        )

    # Decompose K7
    decomposed = transpile(
        payload,
        basis_gates=NATIVE_GATES,
        optimization_level=3,
        seed_transpiler=8,
    )

    # Let QuantEM choose protection
    compiler = QEDCompiler(
    clifford_threshold=0.4,
    default_num_checks=1,
    verbose=False
    
    )
    result = compiler.compile(
        decomposed,
        strategy=QEDStrategy.AUTO,
        num_checks=1
    )

    protected = result.circuit

    # Measure Iceberg bottom qubit for decoding
    b_readout = ClassicalRegister(1, "b_readout")
    protected.add_register(b_readout)

    regs = result.metadata["register_bundle"]
    protected.measure(
        regs.b[0],
        b_readout[0]
    )

    # Convert protected circuit to our native gate set
    protected_native = transpile(
        protected,
        basis_gates=NATIVE_GATES,
        optimization_level=3,
        seed_transpiler=8,
    )

    iceberg_circuits[(a, b)] = protected_native

print("Iceberg circuits built:", len(iceberg_circuits))

iceberg_win_rates = []

iceberg_simulator = AerSimulator(
    noise_model=noise_model,
    method="matrix_product_state"
)

for a, b in input_pairs:

    circuit = iceberg_circuits[(a, b)]

    job = iceberg_simulator.run(
        circuit,
        shots=1024,
        seed_simulator=42
    )

    counts = job.result().get_counts()

    accepted_shots = 0
    winning_shots = 0

    for bitstring, frequency in counts.items():

        parts = bitstring.split()

        b_bit = parts[0]
        logical_bits = parts[1]

        # Reject if Iceberg detected an error
        if any("1" in block for block in parts[2:]):
            continue

        accepted_shots += frequency

        # Decode Iceberg output
        if b_bit == "1":
            logical_bits = "".join(
                "1" if bit == "0" else "0"
                for bit in logical_bits
            )

        bob_index = int(logical_bits[:3], 2)
        alice_index = int(logical_bits[3:], 2)

        if alice_index >= 6 or bob_index >= 6:
            continue

        alice_edge = vertex_edges[a][alice_index]
        bob_edge = vertex_edges[b][bob_index]

        if game_result(alice_edge, bob_edge) == 1:
            winning_shots += frequency

    win_probability = winning_shots / accepted_shots

    iceberg_win_rates.append(win_probability)

    print(
        f"Iceberg ({a}, {b}): "
        f"{win_probability:.2%} "
        f"({accepted_shots} accepted)"
    )

#print boxplot of winning probabilities
plt.figure(figsize=(8, 5))

plt.boxplot([
    [100 * rate for rate in win_rates],
    [100 * rate for rate in iceberg_win_rates]
])

plt.xticks(
    [1, 2],
    ["Decomposed K7", "Decomposed K7 + QuantEM"]
)

plt.ylabel("Winning percentage (%)")
plt.title("K7: Baseline vs QuantEM (p = 0.001)")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("k7_comparison_boxplot.png", dpi=300)
plt.show()