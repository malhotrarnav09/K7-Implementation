import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.providers.basic_provider import BasicSimulator

from src.K7.bases import bases
from src.K7.verifier import vertex_edges, game_result


#create the 7 8x8 matrices
def matrix(m):
  eight_m = np.identity(8, dtype=complex)
  eight_m[:6, :6] = m
  return eight_m

bases_eight = [matrix(m) for m in bases]


def alicestrategy(vertice):
  matrix = bases_eight[vertice-1]
  gate = UnitaryGate(matrix)
  return gate

def bobstrategy(vertice):
  matrix = bases_eight[vertice-1]
  gate = UnitaryGate(np.conjugate(matrix))
  return gate

#the actual circuit based on two inputs
def build_game_circuit(a, b):
  # Create Alice's and Bob's six-qubit register.
  circuit = QuantumCircuit(6)

  # Prepare Alice's six-state input.
  circuit.initialize(
    [1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 2],
    normalize=True
  )

  # Correlate Alice's three qubits with Bob's three qubits.
  circuit.cx(0, 3)
  circuit.cx(1, 4)
  circuit.cx(2, 5)

  # Select the basis determined by each player's vertex.
  agate = alicestrategy(a)
  bgate = bobstrategy(b)

  # Apply the two selected basis transformations.
  circuit.append(agate, [0, 1, 2])
  circuit.append(bgate, [3, 4, 5])

  # Return the circuit before measurement or simulation.
  return circuit

def game_circuit(a, b):
  circuit = build_game_circuit(a, b)

  circuit.measure_all()

#actually measure it
  backend = BasicSimulator()
  circuit = transpile(circuit, backend)
  result = backend.run(circuit, shots=1).result()
#get the measurements as a string
  string = list(result.get_counts())[0]
#separate the first three and last three elements in the string above
  bobstring = string[:3]
  alicestring = string[3:]
#convert the string from base 2 to base 10 
  aliceindex = int(alicestring[0])*4+int(alicestring[1])*2+int(alicestring[2])
  bobindex = int(bobstring[0])*4+int(bobstring[1])*2 + int(bobstring[2])
#find the related edge based on the index 
  aliceedge = vertex_edges[a][aliceindex]
  bobedge = vertex_edges[b][bobindex]
#return the edges that alice and bob input into the game verifier
  return aliceedge, bobedge