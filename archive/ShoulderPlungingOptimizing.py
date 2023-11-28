import time
import numpy as np
from scipy.optimize import fsolve
from sympy import symbols, lambdify
from multiprocessing import Pool

# Define the objective function
def objective_function(params):
    R1, R2, L, Sx, Sy, Sz = params

    def equation(phi, theta):
        val = (R2*np.cos(phi) + Sx)**2 + (R2*np.sin(phi) + Sy - R1*np.sin(theta))**2 + (Sz - R1*np.cos(theta))**2 - L**2
        return val

    theta_values = np.linspace(0, 2*np.pi, 120)
    phi_values = []

    for theta in theta_values:
        phi_solution, = fsolve(equation, 0, args=(theta))
        phi_values.append(phi_solution)
        
    phi_values = np.array(phi_values)

    # Desired curve: sinusoidal function with amplitude 45 centered on the x-axis
    desired_curve = 45 * np.sin(theta_values)
    
    # RMSE calculation
    error = np.sqrt(np.mean((phi_values - desired_curve)**2))
    #print(error)

    return error

# Gradient isn't required for this problem, so I'm skipping it.

# Define the Particle class
class Particle:
    def __init__(self, dimension=6):
        self.position = np.random.uniform(0.1, 50, dimension)
        self.velocity = np.random.uniform(-1, 1, dimension)
        self.best_position = np.copy(self.position)
        self.best_score = objective_function(self.position)

# PSO function
def PSO(hyperparameters, idx, total_combinations):
    start_time = time.time()

    alpha, beta, gamma, delta = hyperparameters
    
    num_particles = 20
    dimension = 6
    
    particles = [Particle(dimension) for _ in range(num_particles)]
    g_best_position = np.random.uniform(0.1, 50, dimension)
    g_best_value = objective_function(g_best_position)

    for _ in range(100):
        for particle in particles:
            value = objective_function(particle.position)
            if value < particle.best_score:
                particle.best_score = value
                particle.best_position = particle.position
            if value < g_best_value:
                g_best_value = value
                g_best_position = particle.position

        for particle in particles:
            inertia = alpha * particle.velocity
            personal_attraction = beta * np.random.random() * (particle.best_position - particle.position)
            global_attraction = gamma * np.random.random() * (g_best_position - particle.position)
            
            particle.velocity = inertia + personal_attraction + global_attraction
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [50, 50, 50, 50, 50, 50])
            if np.any(np.isnan(particle.position)) or np.any(np.isinf(particle.position)):
                particle.position = np.random.uniform(0.1, 30, dimension)

    elapsed_time = time.time() - start_time

    return (alpha, beta, gamma, delta, g_best_value, g_best_position, elapsed_time)

def wrapper(args):
    hyperparameters, idx, total_combinations = args
    result = PSO(hyperparameters, idx, total_combinations)
    
    alpha, beta, gamma, delta, g_best_value, g_best_position, elapsed_time = result

    return (alpha, beta, gamma, delta, g_best_value, g_best_position, elapsed_time)

def grid_search_parallel():
    alphas = [0.1, 0.3, 0.5]
    betas = [0.7, 0.8, 0.9]
    gammas = [0.8, 0.9, 1.0]
    deltas = [0.01, 0.05, 0.1]
    
    combinations = [(alpha, beta, gamma, delta) for alpha in alphas for beta in betas for gamma in gammas for delta in deltas]
    total_combinations = len(combinations)

    results = []

    with Pool() as pool:
        for result in pool.imap_unordered(wrapper, [(combo, idx, total_combinations) for idx, combo in enumerate(combinations)]):
            results.append(result)
            completed = len(results)
            avg_time = sum([r[6] for r in results]) / completed
            estimated_time_left = avg_time * (total_combinations - completed) / pool._processes
            print(f"Completed {completed}/{total_combinations}. Estimated time remaining: {estimated_time_left/60:.2f} minutes.")

    best_hyperparameters = min(results, key=lambda x: x[4])
    return best_hyperparameters

# Main code execution
if __name__ == "__main__":
    best_hyperparameters = grid_search_parallel()

    print(f"Best hyperparameters: alpha={best_hyperparameters[0]}, beta={best_hyperparameters[1]}, gamma={best_hyperparameters[2]}, delta={best_hyperparameters[3]}")
    print(f"Best value achieved: {best_hyperparameters[4]}")
    print(f"Best R1: {best_hyperparameters[5][0]}, Best R2: {best_hyperparameters[5][1]}, Best L: {best_hyperparameters[5][2]}, Best Sx: {best_hyperparameters[5][3]}, Best Sy: {best_hyperparameters[5][4]}, Best Sz: {best_hyperparameters[5][5]}")
