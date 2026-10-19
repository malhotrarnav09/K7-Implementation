from quantem.utils import convert_to_PCS_circ_largest_clifford
from src.K7.quantumcircuit import build_game_circuit
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator
# PCS is Pauli Check Sandwiching. We place pauli checks before and after a circuit block

ALICE_INPUT = 2
BOB_INPUT = 3

NUM_DATA_QUBITS = 6
NUM_CHECKS = 1

original_circuit = build_game_circuit(ALICE_INPUT, BOB_INPUT)

expected_check_signs, circuit_with_pcs = convert_to_PCS_circ_largest_clifford(original_circuit, NUM_DATA_QUBITS, NUM_CHECKS)

#prints final signs of the checks, which should be all +1 if the circuit is correct
print("\nExpected check signs:", expected_check_signs)

#prints the number of qubits in the original circuit and the circuit with PCS
print(
    "Original qubits:",
    original_circuit.num_qubits,
)
print(
    "Qubits in circuit with PCS:",
    circuit_with_pcs.num_qubits,
)

#prints the number of gates in the original circuit and the circuit with PCS
print(
    "Original gate counts:",
    dict(original_circuit.count_ops()),
)

print(
    "Protected gate counts:",
    dict(circuit_with_pcs.count_ops()),
)

#draw the original circuit and the circuit with PCS
print("\nQUANTEM-PROTECTED CIRCUIT\n")

print(
    circuit_with_pcs.draw(
        output="text",
        fold=160,
    )
)

check_bit = ClassicalRegister(1, "check") #creates one classic storage location named check
circuit_with_pcs.add_register(check_bit)

circuit_with_pcs.measure(6,check_bit[0]) #measure the ancila and store the result in check
#we are expecting a value of 0 as the expected sign was +1, which means the ancilla is in the |+> state. applying a hadamard to this state should give us 0 

simulator = AerSimulator()
job = simulator.run(circuit_with_pcs, shots=1024)
result = job.result()

counts = result.get_counts(circuit_with_pcs)
print("\nPCS ancillas counts:", counts)

print("\nNumber of classical bits:" , circuit_with_pcs.num_clbits)

