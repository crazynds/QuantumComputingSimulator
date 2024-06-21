from QuantumSystem import  QuantumCircuit


# Exemplo de uso
qc1 = QuantumCircuit(1)
qc2 = QuantumCircuit(3)

#qc1.apply_hadamard(0)
#qc1.apply_x(0)
#qc1.apply_z(0)

qbit = qc2.teleport(qc1.get_qubit(0))

print(qbit)


# for i in range(N):
#     print(f'{i}#',qc.get_qubit(i))