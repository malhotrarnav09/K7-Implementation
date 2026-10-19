from qiskit import transpile

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from qiskit_aer.noise import ReadoutError

from qiskit_aer.noise import pauli_error

from src.K7.verifier import vertex_edges, game_result

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

simulator = AerSimulator( noise_model=noise_model)

job = simulator.run(
    complete_transpiled,
    shots=1024
)

result = job.result()

counts = result.get_counts(complete_transpiled)

print("\nNoisy K7 measurement counts:")
print(counts)

#calculate winning probability
winning_shots = 0
invalid_shots = 0

for bitstring, frequency in counts.items():

    bob_index = int(bitstring[:3], 2)
    alice_index = int(bitstring[3:], 2)

    # Outcomes 6 and 7 are invalid K7 outputs.
    if alice_index >= 6 or bob_index >= 6:
        invalid_shots += frequency
        continue

    alice_edge = vertex_edges[ALICE_INPUT][alice_index]
    bob_edge = vertex_edges[BOB_INPUT][bob_index]

    if game_result(alice_edge, bob_edge) == 1:
        winning_shots += frequency

total_shots = sum(counts.values())

win_probability = winning_shots / total_shots

print("\nK7 NOISY GAME RESULTS")
print("Winning shots:", winning_shots)
print("Invalid shots:", invalid_shots)
print("Total shots:", total_shots)
print("Winning probability:", win_probability)
print("Winning percentage:", win_probability * 100)
