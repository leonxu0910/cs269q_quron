
from pyquil.quil import address_qubits
from pyquil.api import QVMConnection, WavefunctionSimulator
from pyquil.paulis import PauliSum, PauliTerm, sZ, sI
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from networks import *


qvm = QVMConnection()

sim = WavefunctionSimulator()


def sigmoid(x):
    return np.arctan(np.tan(x)**2)


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

sigmoid_val = sigmoid(np.array(angles)) / (np.pi/2)
plt.plot(angles, percent, 'bo')
plt.plot(angles, sigmoid_val, 'r')
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
