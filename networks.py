from pyquil.gates import MEASURE, I, X, CNOT, H, RY, RZ, Z, RESET
from pyquil.quilatom import QubitPlaceholder
from pyquil import Program
from gates import *

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

