import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Entry, Label, Button, StringVar, DoubleVar, Frame

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
    
    theta_values = np.linspace(0, 2*np.pi, 540)

    def equation(phi, theta):
        val = (R2*np.cos(phi) + Sx)**2 + (R2*np.sin(phi) + Sy - R1*np.sin(theta))**2 + (Sz - R1*np.cos(theta))**2 - L**2
        return val

    phi_values = [fsolve(equation, 0, args=(theta)) for theta in theta_values]
    phi_values_deg = np.degrees(np.array([value[0] for value in phi_values]))
    
    min_phi_value = np.min(phi_values_deg)
    max_phi_value = np.max(phi_values_deg)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(np.degrees(theta_values), phi_values_deg, 'b.', label='Data points')
    #ax.plot(np.degrees(theta_values), np.degrees(sinusoidal(theta_values, A, B, C, D)), label='Sinusoidal Fit')
    
    ax.axhline(y=40, color='r', linestyle='--')
    ax.axhline(y=-30, color='r', linestyle='--')
    
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

window = tk.Tk()
window.title("Interactive Phi vs. Theta")

R1_var = StringVar(value="1.15")
R2_var = StringVar(value="2.4")
L_var = StringVar(value="3.74")
Sx_var = StringVar(value="-2.05")
Sy_var = StringVar(value="2.96")
Sz_var = StringVar(value="1.89")

params = [
    ("R1", R1_var, 0, 0),
    ("R2", R2_var, 0, 3),
    ("L", L_var, 0, 6),
    ("Sx", Sx_var, 2, 0),
    ("Sy", Sy_var, 2, 3),
    ("Sz", Sz_var, 2, 6),
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