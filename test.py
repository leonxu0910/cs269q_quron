import numpy as np
from pyquil import Program
from pyquil.gates import MEASURE, I, X, CNOT, H, RY, RZ, Z, RESET
from pyquil.quil import address_qubits
from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection, WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm, sZ, sI
from scipy.optimize import minimize
import matplotlib.pyplot as plt


qvm = QVMConnection()

sim = WavefunctionSimulator()

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


def rus(inputs, w, b):
    in_reg = list()
    for i in inputs:
        in_reg.append(i)
    an_reg = list()
    an_reg.append(QubitPlaceholder())
    an_reg.append(QubitPlaceholder())
    # Gates and classical memory preparation
    prep_pq = Program()
    acl_ro = prep_pq.declare('acl_{}_ro'.format(an_reg[0]), 'BIT', 1)
    # Rotation gates
    rot_linear_pq = Program()
    for i in range(len(w)):
        rot_linear_pq += CRY(w[i])(in_reg[i], an_reg[0])
    rot_linear_pq += RY(b, an_reg[0])
    rot_pq = Program()
    rot_pq += rot_linear_pq
    rot_pq += CY(an_reg[0], an_reg[1])
    rot_pq += RZ(-np.pi / 2, an_reg[0])
    rot_pq += rot_linear_pq
    # Ancilla bit measurement
    pq = Program()
    pq += prep_pq
    pq += rot_pq
    pq += MEASURE(an_reg[0], acl_ro)
    # Repeated circuit
    rep_pq = Program()
    # rep_pq += RESET(reg[1])
    rep_pq += RY(-np.pi / 2, an_reg[1])
    rep_pq += rot_pq
    rep_pq += MEASURE(an_reg[0], acl_ro)

    pq.while_do(acl_ro, rep_pq)
    return pq, an_reg[1]


def rus_single(input, theta):
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


def xor_networks(inputs, theta):
    w = theta[0:6]
    b = theta[6:len(theta)]
    prep_pq = Program()
    prep_pq += dg_cry
    prep_pq += dg_cy
    hidden_1_pq, hidden_1_reg = rus(inputs, [w[0], w[1]], b[0])
    hidden_2_pq, hidden_2_reg = rus(inputs, [w[2], w[3]], b[1])
    output_pq, output_reg = rus([hidden_1_reg, hidden_2_reg], [w[4], w[5]], b[2])
    pq = prep_pq + hidden_1_pq + hidden_2_pq + output_pq
    return pq, output_reg


angles = [0, (1/8) * np.pi/2, (2/8) * np.pi/2, (3/8) * np.pi/2, (4/8) * np.pi/2, (5/8) * np.pi/2, (6/8) * np.pi/2, (7/8) * np.pi/2, (8/8) * np.pi/2]
percent = []
for a in angles:
    inp = QubitPlaceholder()
    pq, out = rus_single(inp, a)
    # print(address_qubits(pq))
    # pq = rus1(angle)
    pq = Program(X(inp)) + pq
    ro = pq.declare('ro', 'BIT', 1)
    pq += MEASURE(out, ro)
    addressed_pq = address_qubits(pq)
    # print(addressed_pq)
    # y = sim.expectation(addressed_pq, [sZ(0)]).real[0]
    cnt = np.array(qvm.run(addressed_pq, trials=100))
    y = np.sum(cnt.T[0]) / cnt.T[0].shape[0]
    percent.append(y)
print(percent)

plt.plot(angles, percent)
plt.show()


# param = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
# pq, out = xor_networks(inp, param)
# pq = Program(X(inp[0])) + pq
# ro = pq.declare('ro', 'BIT', 1)
# pq += MEASURE(out, ro)
# addressed_pq = address_qubits(pq)
# # print(addressed_pq)
# cnt = np.array(qvm.run(addressed_pq, trials=100))
# y = np.sum(cnt.T[0]) / cnt.T[0].shape[0]
# print(y)


def generate_samples():
    reg_00 = [QubitPlaceholder(), QubitPlaceholder()]
    pq_00 = Program()
    reg_01 = [QubitPlaceholder(), QubitPlaceholder()]
    pq_01 = Program(X(reg_01[1]))
    reg_10 = [QubitPlaceholder(), QubitPlaceholder()]
    pq_10 = Program(X(reg_10[0]))
    reg_11 = [QubitPlaceholder(), QubitPlaceholder()]
    pq_11 = Program(X(reg_11[0]), X(reg_11[1]))
    return [[pq_00, reg_00, -1.0], [pq_01, reg_01, 1.0], [pq_10, reg_10, 1.0], [pq_11, reg_11, -1.0]]


def fun_xor(inputs, param, initial_pq=None):
    pq, reg = xor_networks(inputs, param)
    out_pq = initial_pq + pq
    return out_pq


def fun_rus(inputs, param, initial_pq=None):
    w = param[0: 2]
    b = param[2]
    pq, reg = rus(inputs, w, b)
    initial_pq += dg_cry
    initial_pq += dg_cy
    out_pq = initial_pq + pq
    return out_pq


def train_xor(samples):
    # param = [np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi, np.pi]
    param = [np.pi,0,0,np.pi,np.pi,np.pi,0,0,np.pi]
    s = samples[2]
    sec_reg = QubitPlaceholder()
    val = sim.expectation(address_qubits(fun_xor(s[1], param, s[0])), [sZ(7)]).real[0]
    print(address_qubits(fun_xor(s[1], param, s[0])))
    print(val)
    print(s[2])
    print(abs(val-s[2]))
    # x = sim.expectation(address_qubits(p), [sZ(3)])
    # print(address_qubits(fun_xor(s[1], sec_reg, s[2], param, s[0])))
    # x = sim.expectation(address_qubits(fun_xor(s[1], s[3], x, s[0])), )
    # for s in samples:
    fun = lambda x: abs(sim.expectation(address_qubits(fun_xor(s[1], param, s[0])), [sZ(7)]).real[0] - s[2])
    # fun = lambda x: sim.expectation(address_qubits(fun_xor(s[1], sec_reg, s[2], x, s[0]), [sZ(8)*sZ(0)]))
    res = minimize(fun, np.array(param), method="Nelder-Mead", tol=10**-6)
    return res.x, res.fun


def train_xor_single(samples):
    param = [np.pi, np.pi, np.pi]
    # param = [1.21664167e+00, 1.18965062e+00, 1.18263409e+00, 1.22122501e+00, 1.14936869e+00, 1.21845866e+00, 8.84375255e-05, 8.41587697e-05, 8.84375255e-05]
    s = samples[0]
    val = sim.expectation(address_qubits(fun_rus(s[1], param, s[0])), [sZ(3)]).real[0]
    print(address_qubits(fun_rus(s[1], param, s[0])))
    print(val)
    print(s[2])
    print(abs(val-s[2]))
    # for s in samples:
    fun = lambda x: abs(sim.expectation(address_qubits(fun_rus(s[1], param, s[0])), [sZ(3)]).real[0] - s[2])
    # fun = lambda x: sim.expectation(address_qubits(fun_xor(s[1], sec_reg, s[2], x, s[0]), [sZ(8)*sZ(0)]))
    res = minimize(fun, np.array(param), method="Nelder-Mead", tol=10**-6)
    return res.x, res.fun


# samples = generate_samples()
# print(train_xor(samples))

# q = QubitPlaceholder()
# pq = Program(X(q))
# x = sim.expectation(address_qubits(pq), [sZ(q)])
# print(x)
