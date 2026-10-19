from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.circuit.library import StatePreparation
from quantem.utils import convert_to_PCS_circ

from src.K7.quantumcircuit import build_game_circuit
from draw_k7_decomposed import NATIVE_GATES

from quantem import QEDCompiler, QEDStrategy

from qiskit_aer import AerSimulator

from src.K7.verifier import vertex_edges, game_result

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
    pauli_error,
)

# Start with one input pair
original = build_game_circuit(2,3)

# Build the full K7 logical circuit using unitary state preparation
payload = QuantumCircuit(6)

payload.append(
    StatePreparation(
        [1, 1, 1, 1, 1, 1, 0, 0],
        normalize=True
    ),
    [0, 1, 2]
)

# Add the rest of the original K7 circuit
for instruction in original.data:

    if instruction.operation.name == "initialize":
        continue

    indices = [
        original.find_bit(q).index
        for q in instruction.qubits
    ]

    payload.append(instruction.operation, indices)

# Decompose the entire unitary
decomposed = transpile(
    payload,
    basis_gates=NATIVE_GATES,
    optimization_level=3,
    seed_transpiler=8,
)

print("Decomposed gate counts:", dict(decomposed.count_ops()))

compiler = QEDCompiler(
    clifford_threshold=0.4,
    default_num_checks=1,
    verbose=False
)

analysis = compiler.analyze_circuit(decomposed)

print("\nQED ANALYSIS")
print(analysis)

result = compiler.compile(
    decomposed,
    strategy=QEDStrategy.AUTO,
    num_checks=1
)

protected = result.circuit

b_readout = ClassicalRegister(1, "b_readout")
protected.add_register(b_readout)

regs = result.metadata["register_bundle"]
protected.measure(regs.b[0], b_readout[0])

#rewrite iceberg using rx,ry,rz,cz instead of rxx,rzz,rzx,cz
protected_native = transpile(
    protected,
    basis_gates=NATIVE_GATES,
    optimization_level=3,
    seed_transpiler=8,
)

print("\nICEBERG NATIVE GATE COUNTS")
print(dict(protected_native.count_ops()))
print("ICEBERG NATIVE DEPTH:", protected_native.depth())

print("\nStrategy used:", result.strategy_used)
print("Protected circuit qubits:", protected.num_qubits)
print("Protected classical bits:", protected.num_clbits)
print("Protected gate counts:", dict(protected.count_ops()))
print("Protected depth:", protected.depth())
print("Metadata:", result.metadata)

print("\nCLASSICAL REGISTERS")

for creg in protected.cregs:
    print(creg.name, creg.size)

SHOTS = 1024

# Unprotected ideal circuit
unprotected = decomposed.copy()
unprotected.measure_all()

#noise model
P = 0.001

noise_model = NoiseModel()

noise_model.add_all_qubit_quantum_error(
    depolarizing_error(16 * P / 15, 2),
    ["cx"]
)

single_error = depolarizing_error(
    4 * (P / 10) / 3,
    1
)

noise_model.add_all_qubit_quantum_error(
    single_error,
    ["rx", "ry", "rz"]
)

noise_model.add_all_qubit_readout_error(
    ReadoutError([
        [1 - 5 * P, 5 * P],
        [5 * P, 1 - 5 * P]
    ])
)

noise_model.add_all_qubit_quantum_error(
    pauli_error([
        ("X", 2 * P),
        ("I", 1 - 2 * P)
    ]),
    ["reset"]
)

# Iceberg-protected ideal circuit
protected_counts = AerSimulator(
    method="matrix_product_state"
).run(
    protected,
    shots=SHOTS,
    seed_simulator=42
).result().get_counts()

# Run Iceberg circuit with SI1000-style noise
noisy_counts = AerSimulator(
    noise_model=noise_model,
    method="matrix_product_state"
).run(
    protected_native,
    shots=SHOTS,
    seed_simulator=42
).result().get_counts()

print("\nNOISY ICEBERG SHOTS:", sum(noisy_counts.values()))


accepted_shots = 0
winning_shots = 0

for bitstring, frequency in noisy_counts.items():
    parts = bitstring.split()

    b = parts[0]
    p_bits = parts[1]

    # Reject the shot if any Iceberg check detected an error
    if any("1" in block for block in parts[2:]):
        continue

    accepted_shots += frequency

    # Decode the Iceberg logical output
    if b == "1":
        p_bits = "".join(
            "1" if bit == "0" else "0"
            for bit in p_bits
        )

    bob_index = int(p_bits[:3], 2)
    alice_index = int(p_bits[3:], 2)

    # 6 and 7 are unused K7 outcomes
    if bob_index >= 6 or alice_index >= 6:
        continue

    alice_edge = vertex_edges[2][alice_index]
    bob_edge = vertex_edges[3][bob_index]

    if game_result(alice_edge, bob_edge) == 1:
        winning_shots += frequency

print("\nNOISY ICEBERG RESULTS")
print("Accepted shots:", accepted_shots)
print("Acceptance rate:", accepted_shots / SHOTS)
print("K7 win rate:", winning_shots / accepted_shots)
