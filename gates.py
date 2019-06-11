from pyquil.parameters import Parameter, quil_sin, quil_cos
from pyquil.quilbase import DefGate
import numpy as np
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