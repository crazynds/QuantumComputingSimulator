import numpy as np
from fractions import Fraction
from Qubit import Qubit

class QuantumCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def apply_hadamard(self, qubit_index):
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        # invert qubit index
        qubit_index = self.num_qubits - qubit_index - 1
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
        # invert qubit index
        control_qubit = self.num_qubits - control_qubit - 1
        target_qubit = self.num_qubits - target_qubit - 1

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
    

    def apply_ccnot(self, controll_bits:list, target_qubit):
        size = 2**self.num_qubits
        # invert qubit index
        controll_bits = [self.num_qubits - bit - 1 for bit in controll_bits]
        target_qubit = target_qubit

        U_CCNOT = np.eye(size, dtype=complex)
    
        for i in range(size):
            binary = np.binary_repr(i, width=self.num_qubits)
            if all(binary[c] == '1' for c in controll_bits):
                target_state = i ^ (1 << target_qubit)
                U_CCNOT[i, i] = 0
                U_CCNOT[i, target_state] = 1
                U_CCNOT[target_state, target_state] = 0
                U_CCNOT[target_state, i] = 1
        
        self.state = np.dot(U_CCNOT, self.state)

    

    def apply_x(self, qubit_index):
        X = np.array([[0, 1], [1, 0]])
        # invert qubit index
        qubit_index = self.num_qubits - qubit_index - 1
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
    
    def apply_z(self, qubit_index):
        Z = np.array([[1, 0], [0, -1]])
        for i in range(self.num_qubits):
            if i == qubit_index:
                operator = Z
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
        prob_one = np.sum(np.abs(self.state[one_indices])**2)

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
    
    
    def __str__(self) -> str:
        st = ''
        states = self.get_state()
        for i in range(len(states)):
            state = ''.join(['1' if (i>>(self.num_qubits - 1 - j))&1==1 else '0' for j in range(self.num_qubits)])
            prob = f'{states[i][0]}' if states[i][1]== 0 else f'{states[i][0]}+{states[i][1]}i' if states[i][1]>0 else f'{states[i][0]}-{states[i][1]}i'
            st += f"({prob}\t|{state}])\t"
        return st
    
    def get_qubit(self, qubit_index):
        return Qubit(self, qubit_index)
    
    def teleport(self, state_to_teleport: Qubit):
        # Inicializa três qubits
        state_to_teleport = state_to_teleport.get_state()
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        # Passo 1: Colocar o primeiro qubit no estado a ser teleportado
        self.state[0] = state_to_teleport[0]
        self.state[1] = state_to_teleport[1]

        # Passo 2: Criar um par de Bell entre os qubits 1 e 2
        self.apply_hadamard(1)
        self.apply_cnot(1, 2)
        # Qubit 1 = Alice
        # Qubit 2 = Bob

        # Passo 3: Aplicação das operações de medição no qubit de Alice e no qubit emaranhado
        self.apply_cnot(0, 1)
        self.apply_hadamard(0)

        # Passo 5: Medição dos qubits de Alice
        m0 = self.measure_qubit(0)
        m1 = self.measure_qubit(1)

        # Alice envia os resultados da medição para Bob via canal de comunicação classico

        # Passo 6: Aplicação das operações condicionais em Bob com base nos resultados da medição de Alice
        if m1 == 1:
            self.apply_x(2)
        if m0 == 1:
            self.apply_z(2)

        return self.get_qubit(2)




