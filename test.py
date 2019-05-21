import numpy as np
from pyquil import Program
from pyquil.gates import MEASURE, I, X, CNOT, H, RY, RZ
from pyquil.quil import address_qubits
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection


qvm = QVMConnection()

theta = Parameter('theta')

cry = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, quil_cos(theta / 2), -quil_sin(theta / 2)],
                [0, 0, quil_sin(theta / 2), quil_cos(theta / 2)]])

crx = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, quil_cos(theta / 2), -1j * quil_sin(theta / 2)],
                [0, 0, -1j * quil_sin(theta / 2), quil_cos(theta / 2)]])

crz = np.array([[1, 0, 0,  0],
                [0, 1, 0,  0],
                [0, 0, quil_cos(theta / 2) - 1j * quil_sin(theta / 2), 0],
                [0, 0, 0, quil_cos(theta / 2) + 1j * quil_sin(theta / 2)]])

cy = np.array([[1, 0, 0,  0],
               [0, 1, 0,  0],
               [0, 0, 0, 0 - 1j],
               [0, 0, 0 + 1j, 0]])

dg_cry = DefGate("CRY", cry, [theta])
dg_crx = DefGate("CRX", crx, [theta])
dg_crz = DefGate("CRZ", crz, [theta])
dg_cy = DefGate("CY", cy)

CRY = dg_cry.get_constructor()
CRX = dg_crx.get_constructor()
CRZ = dg_crz.get_constructor()
CY = dg_cy.get_constructor()


def rus(input, theta):
    reg = list()
    reg.append(input)
    reg.append(QubitPlaceholder())
    reg.append(QubitPlaceholder())
    pq = Program()
    pq += CRY(2 * theta)(reg[0], reg[1])
    pq += CY(reg[1], reg[2])
    pq += RZ(-np.pi / 2, reg[1])
    pq += CRY(2 * theta)(reg[0], reg[1])
    return pq, reg


inp = QubitPlaceholder()
angle = 0.1
pq, reg = rus(inp, angle)
print(address_qubits(pq))
addressed_pq = address_qubits(pq).measure_all()
cnt = qvm.run(addressed_pq, trials=10)
print(cnt)

