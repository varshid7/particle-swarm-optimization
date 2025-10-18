import numpy as np
import matplotlib.pyplot as plt

# Objective function (Sphere function example)
def objective(x):
    return np.sum(x**2)

# Particle Swarm Optimization (PSO)
def PSO(obj_func, dim, bounds, num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    # Initialize particles
    lb, ub = bounds
    X = np.random.uniform(lb, ub, (num_particles, dim))  # positions
    V = np.zeros((num_particles, dim))                   # velocities
    pbest = X.copy()                                     # personal best
    pbest_val = np.array([obj_func(x) for x in X])       # personal best value
    gbest = pbest[np.argmin(pbest_val)]                  # global best
    gbest_val = np.min(pbest_val)

    # To store convergence curve
    convergence_curve = []

    # Iterations
    for _ in range(max_iter):
        for i in range(num_particles):
            # Update velocity
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = (w * V[i] +
                    c1 * r1 * (pbest[i] - X[i]) +
                    c2 * r2 * (gbest - X[i]))

            # Update position with bounds
            X[i] = X[i] + V[i]
            X[i] = np.clip(X[i], lb, ub)

            # Evaluate new position
            val = obj_func(X[i])
            if val < pbest_val[i]:
                pbest[i] = X[i].copy()
                pbest_val[i] = val
                if val < gbest_val:
                    gbest = X[i].copy()
                    gbest_val = val

        convergence_curve.append(gbest_val)

    return gbest, gbest_val, convergence_curve

# Example run
dim = 5
bounds = (-5, 5)
best_pos, best_val, convergence = PSO(objective, dim, bounds)

print("Best Position:", best_pos)
print("Best Value:", best_val)

# Plot convergence curve
plt.plot(convergence, label="PSO Convergence")
plt.xlabel("Iteration")
plt.ylabel("Best Objective Value")
plt.title("PSO Convergence Curve")
plt.legend()
plt.grid(True)
plt.show()



