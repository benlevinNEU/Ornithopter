import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Entry, Label, Button, StringVar, DoubleVar, Frame
from sympy import symbols, Eq, solve

N_SMP = 120

def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

def adjust_value(var, amount):
    current_value = float(var.get())
    new_value = round(current_value + amount, 2)  # Round to 2 decimal places
    var.set(new_value)

def update_plot(event=None):
    R1 = float(R1_var.get())
    R2 = float(R2_var.get())
    L = float(L_var.get())
    Sx = float(Sx_var.get())
    Sy = float(Sy_var.get())
    Sz = float(Sz_var.get())

    HCD = float(HCD_var.get())
    AoA = float(AoA_var.get())
    SA = float(SA_var.get())
    SL = float(SL_var.get())
    ED = float(ED_var.get())
    EC = float(EC_var.get())
    CD = float(CD_var.get())
    DPz = float(Cz_var.get())

    Cx = float(Cx_var.get())
    Cy = float(Cy_var.get())
    Cz = float(Cz_var.get())
    
    # Solve for phi

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
    CPx = [Qx(si) + SL * np.cos(beta) for theta, si, beta in beta_values]
    CPy = [Qy(si) + SL * np.sin(beta) for theta, si, beta in beta_values]

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
        eq1 = (Sx + HCD * np.cos(phi) + AoA*(-HCD*np.sin(phi)*np.cos(alpha)) - DPx_val)**2 
        eq2 = (Sy + HCD * np.cos(phi) + AoA*(HCD*np.cos(phi)*np.cos(alpha)) - DPy_val)**2
        eq3 = (Sz + AoA*np.sin(alpha) - DPz_val)**2
        return eq1 + eq2 + eq3 - EC**2

    alpha_values = []
    for phi in phi_values:
        for dp_x, dp_y in zip(DPx, DPy):  # I'm assuming DPz can be computed or is provided
            alpha_sol = fsolve(equation_for_alpha, 0, args=(phi, dp_x, dp_y, DPz))
            alpha_values.append(alpha_sol[0])

    alpha_values = np.array(alpha_values)

    #(Sx + HCD * np.cos(phi_values) + AoA*(-HCD*np.sin(phi_values)*np.cos(alpha)) - DPx)**2 + (Sy + HCD * np.cos(phi_values) + AoA*(HCD*np.cos(phi_values)*np.cos(alpha)) - DPx)**2 + (Sz + AoA*np.sin(alpha) - DPz)**2 == EC**2

    min_phi_value = np.min(phi_values_deg)
    max_phi_value = np.max(phi_values_deg)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(np.degrees(theta_values), phi_values_deg, 'b.', label='Data points')
    #ax.plot(np.degrees(theta_values), np.degrees(sinusoidal(theta_values, A, B, C, D)), label='Sinusoidal Fit')
    
    ax.axhline(y=45, color='r', linestyle='--')
    ax.axhline(y=-45, color='r', linestyle='--')
    
    # Displaying the min and max phi values on the plot
    ax.text(-10, min_phi_value, f"Min Phi: {min_phi_value:.2f}°      ", color='red', verticalalignment='bottom', horizontalalignment='right')
    ax.text(-10, max_phi_value, f"Max Phi: {max_phi_value:.2f}°      ", color='red', verticalalignment='top', horizontalalignment='right')
    
    # Determine the theta values at which min_phi and max_phi occur
    min_phi_theta = np.degrees(theta_values[np.argmin(phi_values_deg)])
    max_phi_theta = np.degrees(theta_values[np.argmax(phi_values_deg)])

    # Add vertical lines to show where the min and max phi are on the theta axis
    ax.axvline(x=min_phi_theta, color='b', linestyle='--')
    ax.axvline(x=max_phi_theta, color='b', linestyle='--')

    # Annotate these lines with the corresponding phi values
    ax.text(min_phi_theta + 18, -55, f"{min_phi_theta:.2f}°", color='blue', verticalalignment='bottom', horizontalalignment='center')
    ax.text(max_phi_theta + 18, -55, f"{max_phi_theta:.2f}°", color='blue', verticalalignment='bottom', horizontalalignment='center')

    downStroke = abs(max_phi_theta - min_phi_theta) / 360
    upStroke = 1- downStroke

    ax.text(270, 55, f"Downstroke Time Prop: {downStroke:.2f}      ", color='black', verticalalignment='bottom', horizontalalignment='left')
    ax.text(270, 55, f"Upstroke Time Prop: {upStroke:.2f}      ", color='black', verticalalignment='top', horizontalalignment='left')

    ax.set_xlabel('Theta (degrees)')
    ax.set_ylabel('Phi (degrees)')
    #ax.legend()
    ax.set_title('Phi vs. Theta')
    ax.set_ylim([-60, 60])
    ax.grid(True)
    plt.close(fig)
    
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=5, column=0, columnspan=9)

    fig3D = plt.figure(figsize=(10, 6))
    ax3D = fig3D.add_subplot(111, projection='3d')
    
    # Extract unique theta, si and alpha values for 3D plotting
    unique_theta_values = np.unique([t for t, s, b in beta_values])
    unique_si_values = np.unique([s for t, s, b in beta_values])

    theta_grid, si_grid = np.meshgrid(unique_theta_values, unique_si_values)

    alpha_grid = np.array([alpha_values[(beta_values[:, 0] == t) & (beta_values[:, 1] == s)][0] for t, s in zip(np.ravel(theta_grid), np.ravel(si_grid))]).reshape(theta_grid.shape)
    
    ax3D.plot_surface(theta_grid, si_grid, alpha_grid, cmap='viridis')
    ax3D.set_xlabel('Theta')
    ax3D.set_ylabel('Si')
    ax3D.set_zlabel('Alpha')
    ax3D.set_title('Alpha vs. Theta & Si')
    plt.close(fig3D)

    canvas3D = FigureCanvasTkAgg(fig3D, master=window)
    canvas3D.draw()
    canvas3D.get_tk_widget().grid(row=6, column=0, columnspan=9) # Assuming this to be the next row after your 2D plot

window = tk.Tk()
window.title("Interactive Phi vs. Theta")

R1_var = StringVar(value="1.4")
R2_var = StringVar(value="2.4")
L_var = StringVar(value="3.63")
Sx_var = StringVar(value="-2.05")
Sy_var = StringVar(value="2.96")
Sz_var = StringVar(value="1.82")

HCD_var = StringVar(value="0.0")
AoA_var = StringVar(value="0.0")
SA_var = StringVar(value="0.0")
SL_var = StringVar(value="0.0")
DPz_var = StringVar(value="0.0")

ED_var = StringVar(value="0.0")
EC_var = StringVar(value="0.0")
CD_var = StringVar(value="0.0")

Cx_var = StringVar(value="0.0")
Cy_var = StringVar(value="0.0")
Cz_var = StringVar(value="0.0")

params = [
    ("R1", R1_var, 0, 0),
    ("R2", R2_var, 0, 3),
    ("L", L_var, 0, 6),
    ("Sx", Sx_var, 2, 0),
    ("Sy", Sy_var, 2, 3),
    ("Sz", Sz_var, 2, 6),
    ("HCD", HCD_var, 8, 0),
    ("AoA", AoA_var, 8, 3),
    ("SA", SA_var, 10, 0),
    ("SL", SL_var, 10, 3),
    ("DPz", DPz_var, 10, 6),
    ("ED", ED_var, 12, 0),
    ("EC", EC_var, 12, 3),
    ("CD", CD_var, 12, 6),
    ("Cx", Cx_var, 14, 0),
    ("Cy", Cy_var, 14, 3),
    ("Cz", Cz_var, 14, 6)
]

for label_text, var, row, col in params:
    Label(window, text=label_text).grid(row=row, column=col)
    entry = Entry(window, textvariable=var, width=5)
    entry.grid(row=row, column=col+1)
    entry.bind('<Return>', update_plot)
    
    btn_frame = Frame(window)
    btn_frame.grid(row=row, column=col+2)
    
    Button(btn_frame, text="↑", command=lambda var=var: (adjust_value(var, 0.01), update_plot())).pack(side='top')
    Button(btn_frame, text="↓", command=lambda var=var: (adjust_value(var, -0.01), update_plot())).pack(side='bottom')

window.after(10, update_plot)
window.mainloop()