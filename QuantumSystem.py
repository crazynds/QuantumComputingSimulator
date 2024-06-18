import numpy as np
from fractions import Fraction

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply_hadamard(self, qubit_index):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for i in range(self.num_qubits):
            if i == qubit_index:
                operator = H
            else:
                operator = np.eye(2)
            if i == 0:
                full_operator = operator
            else:
                full_operator = np.kron(full_operator, operator)
        self.state = np.dot(full_operator, self.state)

    def apply_cnot(self, control_qubit, target_qubit):
        size = 2**self.num_qubits
        CNOT = np.zeros((size, size), dtype=complex)
        for i in range(size):
            binary = np.binary_repr(i, width=self.num_qubits)
            if binary[control_qubit] == '0':
                j = i
            else:
                flipped = binary[:target_qubit] + ('1' if binary[target_qubit] == '0' else '0') + binary[target_qubit+1:]
                j = int(flipped, 2)
            CNOT[i, j] = 1
        self.state = np.dot(CNOT, self.state)
    
    def apply_x(self, qubit_index):
        X = np.array([[0, 1], [1, 0]])
        for i in range(self.num_qubits):
            if i == qubit_index:
                operator = X
            else:
                operator = np.eye(2)
            if i == 0:
                full_operator = operator
            else:
                full_operator = np.kron(full_operator, operator)
        self.state = np.dot(full_operator, self.state)

    def measure_all(self):
        probabilities = np.abs(self.state)**2
        result = np.random.choice(range(2**self.num_qubits), p=probabilities)
        binary_result = np.binary_repr(result, width=self.num_qubits)
        return binary_result

    def measure_qubit(self, qubit_index):
        size = 2**self.num_qubits
        zero_indices = []
        one_indices = []
        
        for i in range(size):
            if (i >> qubit_index) % 2 == 0:
                zero_indices.append(i)
            else:
                one_indices.append(i)

        prob_zero = np.sum(np.abs(self.state[zero_indices])**2)
        prob_one = np.sum(np.abs(self.state[one_indices]))**2

        result = np.random.choice([0, 1], p=[prob_zero, prob_one])

        if result == 0:
            self.state[one_indices] = 0
            self.state /= np.sqrt(prob_zero)
        else:
            self.state[zero_indices] = 0
            self.state /= np.sqrt(prob_one)
        
        return result

    def get_state(self):
        state_fractions = []
        for amplitude in self.state:
            real_part = Fraction(amplitude.real).limit_denominator(100).as_integer_ratio()
            imag_part = Fraction(amplitude.imag).limit_denominator(100).as_integer_ratio()
            state_fractions.append([int(a[0]/a[1]) if a[0]%a[1]==0 else ('/'.join(map(str, a)) if a[0] != 0 else '0') for a in [real_part, imag_part]])
        return state_fractions

# Exemplo de uso
qc = QuantumCircuit(4)
print("Estado inicial:")
print(qc.get_state())

qc.apply_hadamard(2)

qc.apply_x(0)
print("\nAplicando a porta Pauli-X:")
print(qc.get_state())