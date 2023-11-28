import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Entry, Label, Button, StringVar, DoubleVar, Frame
from sympy import symbols, Eq, solve, lambdify

N_SMP = 120

R1 = 1.4
R2 = 2.4
L = 3.63
Sx = -2.05
Sy = 2.96
Sz = 1.82

HCD = 20
AoA = 10
SA = 33
SL = 36
ED = 36
EC = 36
CD = 52
DPz = 41.42

Cx = 31.7
Cy = 2.4
Cz = 0

def generateTable():

    theta_values = np.linspace(0, 2*np.pi, N_SMP)

    def equationPhiTheta(phi, theta):
        val = (R2*np.cos(phi) + Sx)**2 + (R2*np.sin(phi) + Sy - R1*np.sin(theta))**2 + (Sz - R1*np.cos(theta))**2 - L**2
        return val

    phi_values = [fsolve(equationPhiTheta, 0, args=(theta)) for theta in theta_values]
    phi_values_deg = np.degrees(np.array([value[0] for value in phi_values]))

    # Solve for beta

    Qx = lambda si: Cx + SA * np.cos(si)
    Qy = lambda si: Cy + SA * np.sin(si)
    Px = lambda theta: R1 * np.cos(theta)
    Py = lambda theta: R1 * np.sin(theta)

    # Solve for angle from origin to (Cx,Cy) and make sure only half those angle are tested (prevents bistability)
    si_bounds = np.arctan2(Cy, Cx)
    si_values = np.linspace(si_bounds-np.pi, si_bounds, int(N_SMP/2))

    def solveForBeta(beta, theta, si):
        val = (Qx(si)+ SL * np.cos(beta) - Px(theta))**2 + (Qy(si) + SL * np.sin(beta) - Py(theta))**2 - ED**2
        return val
    
    beta_values = []
    for theta in theta_values:
        for si in si_values:
            beta = fsolve(solveForBeta, 0, args=(theta, si))
            beta_values.append((theta, si, beta[0]))

    beta_values = np.array(beta_values)

   # Computation for CPx and CPy
    CPx = [(theta, si, Qx(si) + SL * np.cos(beta)) for theta, si, beta in beta_values]
    CPy = [(theta, si, Qy(si) + SL * np.sin(beta)) for theta, si, beta in beta_values]

    # Convert lists to numpy arrays
    CPx = np.array(CPx)
    CPy = np.array(CPy)

    DPx, DPy = symbols('DPx DPy')

    # Define the two circle equations
    eq1 = Eq((DPx - Px(theta_values))**2 + (DPy - Py(theta_values))**2, EC**2)
    eq2 = Eq((DPx - CPx)**2 + (DPy - CPy)**2, CD**2)

    

    # Solve the simultaneous equations
    solutions = solve((eq1,eq2), (DPx, DPy))

    # AoA attachment point parametric equations
    Xc = lambda phi, alpha: Sx + HCD * np.cos(phi) + AoA * (-HCD * np.sin(phi) * np.cos(alpha))
    Yc = lambda phi, alpha: Sy + HCD * np.cos(phi) + AoA * (HCD * np.cos(phi) * np.cos(alpha))
    Zc = lambda phi, alpha: Sz + AoA * np.sin(alpha)

    # Solve for alpha
    def equation_for_alpha(alpha, phi, DPx_val, DPy_val, DPz_val):
        eq1 = (Sx + HCD * np.cos(phi) + AoA * (-HCD * np.sin(phi) * np.cos(alpha)) - DPx_val)**2 
        eq2 = (Sy + HCD * np.cos(phi) + AoA * (HCD * np.cos(phi) * np.cos(alpha)) - DPy_val)**2
        eq3 = (Sz + AoA * np.sin(alpha) - DPz_val)**2
        return eq1 + eq2 + eq3 - EC**2

    alpha_values = []
    for phi in phi_values:
        for dp_x, dp_y in zip(DPx, DPy):  # I'm assuming DPz can be computed or is provided
            alpha_sol = fsolve(equation_for_alpha, 0, args=(phi, dp_x, dp_y, DPz))
            alpha_values.append(alpha_sol[0])

    alpha_values = np.array(alpha_values)
    return alpha_values

print(generateTable())