import numpy as np
from pyquil import Program
from pyquil.gates import MEASURE, I, X, CNOT, H, RY, RZ, RESET
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
# print(type(dg_cry))
# print(dg_cry)

CRY = dg_cry.get_constructor()
CRX = dg_crx.get_constructor()
CRZ = dg_crz.get_constructor()
CY = dg_cy.get_constructor()


def rus(input, theta):
    reg = list()
    reg.append(input)
    reg.append(QubitPlaceholder())
    reg.append(QubitPlaceholder())
    # Gates and classical memory preparation
    prep_pq = Program()
    prep_pq += dg_cry
    prep_pq += dg_cy
    acl_ro = prep_pq.declare('acl_ro', 'BIT', 1)
    # Rotation gates
    rot_pq = Program()
    rot_pq += CRY(2 * theta)(reg[0], reg[1])
    rot_pq += CY(reg[1], reg[2])
    rot_pq += RZ(-np.pi / 2, reg[1])
    rot_pq += CRY(2 * theta)(reg[0], reg[1])
    # Ancilla bit measurement
    pq = Program()
    pq += prep_pq
    pq += rot_pq
    pq += MEASURE(reg[1], acl_ro)
    # Repeated circuit
    rep_pq = Program()
    # rep_pq += RESET(reg[1])
    rep_pq += RY(-np.pi / 2, reg[2])
    rep_pq += rot_pq
    rep_pq += MEASURE(reg[1], acl_ro)

    pq.while_do(acl_ro, rep_pq)
    return pq, reg[2]


inp = QubitPlaceholder()
angle = np.pi/2
pq, out = rus(inp, angle)
# print(address_qubits(pq))
# pq = rus1(angle)
pq = Program(H(inp)) + pq
ro = pq.declare('ro', 'BIT', 1)
pq += MEASURE(out, ro)
addressed_pq = address_qubits(pq)
print(addressed_pq)
cnt = np.array(qvm.run(addressed_pq, trials=100))
y = np.sum(cnt.T[0]) / cnt.T[0].shape[0]
print(y)


