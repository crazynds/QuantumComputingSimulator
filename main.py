from QuantumSystem import  QuantumCircuit


# Exemplo de uso
qc = QuantumCircuit(16)

qc.apply_hadamard(1)
qc.apply_hadamard(0)

qc.apply_x(2)

qc.apply_ccnot([0,1], 2)

print(qc)

