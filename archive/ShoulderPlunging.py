import numpy as np
from scipy.optimize import fsolve, curve_fit
import matplotlib.pyplot as plt

# Define parameters
#R1 = 1.4  # Radius related to theta
#R2 = 2.0  # Radius related to phi
#L = 2.58   # Length of the rod
#Sx = 0.2  # Static x-offset
#Sy = 2  # Static y-offset
#Sz = 0  # Static z-offset

# Define parameters
R1 = 1.4  # Radius related to theta
R2 = 2.3  # Radius related to phi
L = 2.97   # Length of the rod
Sx = 0.2  # Static x-offset
Sy = 2  # Static y-offset
Sz = 0.9  # Static z-offset

# Define parameters
#R1 = 22.7945254268716  # Radius related to theta
#R2 = 1.8159646764830932  # Radius related to phi
#L = 48.430773311410036   # Length of the rod
#Sx = 45.18753955272525  # Static x-offset
#Sy = -3.463025004411898  # Static y-offset
#Sz = 33.61772356869928  # Static z-offset

def equation(phi, theta):
    # Use the provided equation
    val = (R2*np.cos(phi) + Sx)**2 + (R2*np.sin(phi) + Sy - R1*np.sin(theta))**2 + (Sz - R1*np.cos(theta))**2 - L**2
    return val

# Sample 360 values for theta
theta_values = np.linspace(0, 2*np.pi, 240)

# Solve for phi for each theta
phi_values = []
for theta in theta_values:
    phi_solution, = fsolve(equation, 0, args=(theta))
    phi_values.append(phi_solution)

phi_values = np.array(phi_values)
#phi_values = np.clip(phi_values, -np.pi/3, np.pi/3)  # Limit phi to +/- 60 degrees

def sinusoidal(x, A, B, C, D):
    return A * np.sin(B * x + C) + D

# Perform sinusoidal fit
params, covariance = curve_fit(sinusoidal, theta_values, phi_values)
A, B, C, D = params

# Predicted phi values from the sinusoidal fit
phi_predicted = sinusoidal(theta_values, A, B, C, D)

# Calculate R^2 value
ss_res = np.sum((phi_values - phi_predicted) ** 2)
ss_tot = np.sum((phi_values - np.mean(phi_values)) ** 2)
r2_value = 1 - (ss_res / ss_tot)

# Print the sinusoidal function and R^2 value
print(f"Sinusoidal fit: y = {A:.4f} * sin({B:.4f} * x + {C:.4f}) + {D:.4f}")
print(f"R^2 value: {r2_value:.4f}")

# Visualize the results
plt.figure(figsize=(10,6))
plt.plot(np.degrees(theta_values), np.degrees(phi_values), 'b.', label='Data points')
plt.plot(np.degrees(theta_values), np.degrees(sinusoidal(theta_values, A, B, C, D)), label='Sinusoidal Fit')

plt.xlabel('Theta (degrees)')
plt.ylabel('Phi (degrees)')
plt.legend()
plt.title('Phi vs. Theta with Sinusoidal Fit')
plt.ylim([-60, 60])  # Limit the phi axis to +/- 60 degrees
plt.grid(True)
plt.show()
plt.show()
