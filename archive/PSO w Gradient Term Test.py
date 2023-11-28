import numpy as np
from sympy import symbols, lambdify
from multiprocessing import Pool

# Objective Function: Rosenbrock function
def objective_function(position):
    x, y = position
    return (1 - x)**2 + 100*(y - x**2)**2

# Gradient of Objective Function
x, y = symbols('x y')
rosenbrock_expr = (1 - x)**2 + 100*(y - x**2)**2
gradient_expr = [rosenbrock_expr.diff(var) for var in (x, y)]
gradient_func = lambdify((x, y), gradient_expr, 'numpy')

# Particle class
class Particle:
    def __init__(self, dimension):
        self.position = np.random.uniform(-5, 5, dimension)
        self.velocity = np.random.uniform(-1, 1, dimension)
        self.best_position = np.copy(self.position)
        self.best_score = objective_function(self.position)

# PSO function
def PSO(hyperparameters):
    alpha, beta, gamma, delta = hyperparameters
    
    num_particles = 30
    dimension = 2
    
    particles = [Particle(dimension) for _ in range(num_particles)]
    g_best_position = np.random.uniform(-5, 5, dimension)
    g_best_value = objective_function(g_best_position)

    # PSO main loop
    for _ in range(1000):
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
            gradient = np.array(gradient_func(*particle.position))
            gradient_influence = delta * gradient
            particle.velocity = inertia + personal_attraction + global_attraction - gradient_influence
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, -5, 5)
            if np.any(np.isnan(particle.position)) or np.any(np.isinf(particle.position)):
                particle.position = np.random.uniform(-5, 5, dimension)
                
    return (alpha, beta, gamma, delta, g_best_value)

def grid_search_parallel():
    # Grid search hyperparameters
    alphas = [0.1, 0.3, 0.5]
    betas = [0.7, 0.8, 0.9]
    gammas = [0.8, 0.9, 1.0]
    deltas = [0.01, 0.05, 0.1]
    
    combinations = [(alpha, beta, gamma, delta) for alpha in alphas for beta in betas for gamma in gammas for delta in deltas]

    with Pool() as pool:
        results = pool.map(PSO, combinations)

    # Extract best hyperparameters
    best_hyperparameters = min(results, key=lambda x: x[4])

    return best_hyperparameters

if __name__ == "__main__":
    best_hyperparameters = grid_search_parallel()

    print(f"Best hyperparameters: alpha={best_hyperparameters[0]}, beta={best_hyperparameters[1]}, gamma={best_hyperparameters[2]}, delta={best_hyperparameters[3]}")
    print(f"Best value achieved: {best_hyperparameters[4]}")
