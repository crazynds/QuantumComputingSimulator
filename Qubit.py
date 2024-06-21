import numpy as np
from fractions import Fraction

class QuantumCircuit:
    pass

class Qubit:
    def __init__(self, circuit: QuantumCircuit, qubit_index):
        self.circuit = circuit
        self.qubit_index = qubit_index

    def get_state(self):
        total_state = self.circuit.state
        qubit_state = np.zeros(2, dtype=complex)

        for i in range(len(total_state)):
            if (i >> self.qubit_index) % 2 == 0:
                qubit_state[0] += total_state[i]
            else:
                qubit_state[1] += total_state[i]

        normalization_factor = np.linalg.norm(qubit_state)
        if normalization_factor != 0:
            qubit_state /= normalization_factor

        return [qubit_state[0], qubit_state[1]]
    
    def __str__(self) -> str:
        state = self.get_state()
        real_part_0 = Fraction(state[0].real).limit_denominator(100).as_integer_ratio()
        imag_part_0 = Fraction(state[0].imag).limit_denominator(100).as_integer_ratio()
        real_part_1 = Fraction(state[1].real).limit_denominator(100).as_integer_ratio()
        imag_part_1 = Fraction(state[1].imag).limit_denominator(100).as_integer_ratio()

        f = lambda arr: [int(a[0]/a[1]) if a[0]%a[1]==0 else ('/'.join(map(str, a)) if a[0] != 0 else '0') for a in arr]
        state = [f([real_part_0,imag_part_0]), f([real_part_1,imag_part_1])]
        p1 = f'{state[0][0]}' if state[0][1]== 0 else f'{state[0][0]}+{state[0][1]}i' if state[0][1]>0 else f'{state[0][0]}-{state[0][1]}i'
        p2 = f'{state[1][0]}' if state[1][1]== 0 else f'{state[1][0]}+{state[1][1]}i' if state[1][1]>0 else f'{state[1][0]}-{state[1][1]}i'
        return f"({p1}\t|0])\t({p2}\t|1])"
