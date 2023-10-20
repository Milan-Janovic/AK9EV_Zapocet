import time
import numpy as np
import matplotlib.pyplot as plt


def identity(v):
    x = np.linalg.norm(v)
    return x


def linear(v):
    x = np.linalg.norm(v)
    return 2 * x


def parabola(v):
    x = np.linalg.norm(v)
    return x ** 2


def sinus(v):
    x = np.linalg.norm(v)
    return np.sin(x)


def cosinus(v):
    x = np.linalg.norm(v)
    return np.cos(x)


def exponential_growth(v):
    x = np.linalg.norm(v)
    return 2 ** x


def exponential_decay(v):
    x = np.linalg.norm(v)
    return 2 ** (-x)


def sigmoid(v):
    x = np.linalg.norm(v)
    return 1 / (1 + np.exp(-x))


def triangle(v):
    x = np.linalg.norm(v)
    return np.abs(x % 4 - 2) - 1


def quadratic_sine(v):
    x = np.linalg.norm(v)
    return np.sin(x) + 0.5 * np.sin(2 * x)


def gaussian_bell(v):
    x = np.linalg.norm(v)
    return np.exp(-x ** 2)


def sawtooth(v):
    x = np.linalg.norm(v)
    return 2 * (x % 1) - 1


def tanh(v):
    x = np.linalg.norm(v)
    return np.tanh(x)


def logistic_sigmoid(v):
    x = np.linalg.norm(v)
    return 1 / (1 + np.exp(-x))


def quadratic_growth(v):
    x = np.linalg.norm(v)
    return x ** 2 + 1


def quadratic_decay(v):
    x = np.linalg.norm(v)
    return -x ** 2 + 1


def square_root(v):
    x = np.linalg.norm(v)
    return np.sqrt(x)


def cubic(v):
    x = np.linalg.norm(v)
    return x ** 3


def logarithmic(v):
    x = np.linalg.norm(v)
    return np.log(x + 1)


def power(v):
    x = np.linalg.norm(v)
    return x ** 3


def reuleaux_triangle(v):
    x = np.linalg.norm(v)
    return np.maximum.reduce([np.abs(x), np.abs(x - 2), np.abs(x + 2)]) - 1


def dirichlet(v):
    x = np.linalg.norm(v)
    return np.sin(x) / np.where(x == 0, 1, x)  # Avoid division by zero


def logarithmic_growth(v):
    x = np.linalg.norm(v)
    return np.log(x + 1)


def logarithmic_decay(v):
    x = np.linalg.norm(v)
    return -np.log(x + 1)


def inverted_sine(v):
    x = np.linalg.norm(v)
    return -np.sin(x)


def hyperbolic_cosine(v):
    x = np.linalg.norm(v)
    return np.cosh(x)


def plot_function(func):
    filename2D = "2D Contour Plot - " + func.__name__
    filename3D = "3D Surface Plot - " + func.__name__

    x = np.linspace(-100, 100, 1000)
    y = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(x, y)

    # Applying the function element-wise
    Z = np.vectorize(lambda i, j: func(np.array([i, j])))(X, Y)

    # 2D plot
    plt.figure(figsize=(10, 6))
    plt.contourf(X, Y, Z, levels=100, cmap='viridis')
    plt.colorbar()
    plt.title("2D Contour Plot - " + func.__name__)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save to file & clear the plot
    plt.savefig(f"./Plots/{filename2D}.png")
    plt.clf()

    # 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Surface Plot - " + func.__name__)

    # Save to file & clear the plot to not
    plt.savefig(f"./Plots/{filename3D}.png")
    plt.clf()


# Define Array containing all the functions
functions = [
    identity,
    linear,
    parabola,
    sinus,
    cosinus,
    exponential_growth,
    exponential_decay,
    sigmoid,
    triangle,
    quadratic_sine,
    gaussian_bell,
    sawtooth,
    tanh,
    logistic_sigmoid,
    quadratic_growth,
    quadratic_decay,
    square_root,
    cubic,
    logarithmic,
    power,
    reuleaux_triangle,
    dirichlet,
    logarithmic_growth,
    logarithmic_decay,
    inverted_sine,
]

# Create 2D & 3D plots for all the functions
#for test_function in functions:
    #plot_function(test_function)


def differential_evolution(func, D, bounds, FEs, repetitions=30):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    CR = 0.9  # Crossover probability
    F = 0.8  # Differential weight

    results = []

    for _ in range(repetitions):
        # Initialize population
        pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        new_pop = np.empty_like(pop)

        # Store function evaluations
        fitness = np.apply_along_axis(func, 1, pop)
        new_fitness = np.empty_like(fitness)

        for _ in range(int(FEs / D)):
            for i in range(NP):
                # Mutation: rand/1
                a, b, c = pop[np.random.choice(NP, 3, replace=False)]
                mutant = a + F * (b - c)

                # Crossover: bin
                cross_points = np.random.rand(D) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, D)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Reflection for boundary control
                trial = np.where(trial < bounds[:, 0], 2 * bounds[:, 0] - trial, trial)
                trial = np.where(trial > bounds[:, 1], 2 * bounds[:, 1] - trial, trial)

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            # Swap the populations and their fitness values
            pop, new_pop = new_pop, pop
            fitness, new_fitness = new_fitness, fitness

        results.append(fitness.min())

    return results

def differential_evolution_best(func, D, bounds, FEs, repetitions=30):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    CR = 0.9  # Crossover probability
    F = 0.5  # Differential weight

    results = []

    for _ in range(repetitions):
        # Initialize population
        pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        new_pop = np.empty_like(pop)

        # Store function evaluations
        fitness = np.apply_along_axis(func, 1, pop)
        new_fitness = np.empty_like(fitness)

        for _ in range(int(FEs / D)):
            best_idx = np.argmin(fitness)
            best = pop[best_idx]

            for i in range(NP):
                # Mutation: best/1
                a, b = pop[np.random.choice(NP, 2, replace=False)]
                mutant = best + F * (a - b)

                # Crossover: bin
                cross_points = np.random.rand(D) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, D)] = True

                trial = np.where(cross_points, mutant, pop[i])

                # Reflection for boundary control
                trial = np.where(trial < bounds[:, 0], 2 * bounds[:, 0] - trial, trial)
                trial = np.where(trial > bounds[:, 1], 2 * bounds[:, 1] - trial, trial)

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_pop[i] = pop[i]
                    new_fitness[i] = fitness[i]

            # Swap the populations and their fitness values
            pop, new_pop = new_pop, pop
            fitness, new_fitness = new_fitness, fitness

        results.append(fitness.min())

    return results


def pso(func, D, bounds, FEs, repetitions=30, w=0.7298, c1=1.49618, c2=1.49618):
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    # Maximum velocity
    v_max = (bounds[:, 1] - bounds[:, 0]) * 0.1

    results = []
    for _ in range(repetitions):
        positions = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        new_positions = np.empty_like(positions)  # Array to store new positions temporarily
        velocities = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = positions[np.argmin(personal_best_scores)]

        for _ in range(int(FEs / NP)):
            for i in range(NP):
                random_p = np.random.rand(D)
                random_g = np.random.rand(D)

                # Update velocities and clamp them
                velocities[i] = w * velocities[i] + c1 * random_p * (
                            personal_best_positions[i] - positions[i]) + c2 * random_g * (
                                            global_best_position - positions[i])
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update positions and handle boundary conditions using reflection
                new_positions[i] = positions[i] + velocities[i]
                new_positions[i] = np.where(new_positions[i] < bounds[:, 0], 2 * bounds[:, 0] - new_positions[i], new_positions[i])
                new_positions[i] = np.where(new_positions[i] > bounds[:, 1], 2 * bounds[:, 1] - new_positions[i], new_positions[i])

                # Update personal best based on new positions
                if func(new_positions[i]) < personal_best_scores[i]:
                    personal_best_scores[i] = func(new_positions[i])
                    personal_best_positions[i] = new_positions[i]

            # Update main positions with the new ones
            positions = new_positions.copy()

            # Update global best based on new positions
            global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        results.append(func(global_best_position))

    return results


def soma_all_to_one(func, D, bounds, FEs, repetitions=30, PathLength=3, StepSize=0.11, PRT=0.7):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    results = []

    for _ in range(repetitions):
        # Initialize population
        pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        new_population = np.copy(pop)
        fitness_cache = np.array([func(ind) for ind in pop])

        for _ in range(int(FEs / D)):
            # Find the best individual
            leader_index = np.argmin(fitness_cache)
            leader = pop[leader_index]

            # Compute journeys for all individuals in a vectorized manner
            journeys = (leader - pop) * PathLength
            steps = int(PathLength / StepSize)

            # Migrate all individuals towards the leader
            for i in range(NP):
                if i != leader_index:  # Ensure we're not trying to migrate the leader to itself
                    current_pos = pop[i]
                    current_fitness = fitness_cache[i]

                    # Calculate all steps for the current individual
                    for step in range(steps):
                        prt_vector = np.where(np.random.rand(D) < PRT, 1, 0)
                        new_pos = current_pos + StepSize * journeys[i] * prt_vector

                        # Reflection for boundary control
                        new_pos = np.where(new_pos < bounds[:, 0], 2 * bounds[:, 0] - new_pos, new_pos)
                        new_pos = np.where(new_pos > bounds[:, 1], 2 * bounds[:, 1] - new_pos, new_pos)

                        new_fitness = func(new_pos)
                        if new_fitness < current_fitness:
                            current_fitness = new_fitness
                            current_pos = new_pos

                    new_population[i] = current_pos
                    fitness_cache[i] = current_fitness

            # Replace the old population with the new one
            pop = np.copy(new_population)

        results.append(fitness_cache[leader_index])

    return results


def soma_all_to_all(func, D, bounds, FEs, repetitions=30, PathLength=3, StepSize=0.11, PRT=0.7):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    results = []

    for _ in range(repetitions):
        # Initialize population
        pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        fitness_cache = np.array([func(ind) for ind in pop])

        for _ in range(int(FEs / NP)):
            new_population = pop.copy()

            # For each individual in the population
            for i in range(NP):
                # Broadcasted difference between individual i and all others
                differences = pop - pop[i]

                # Calculate journey for all individuals in one go
                journeys = differences * PathLength
                steps = int(PathLength / StepSize)
                current_pos = np.tile(pop[i], (NP, 1))
                current_fitness = fitness_cache[i]

                # Calculate all steps for the current individual
                for step in range(steps):
                    prt_vectors = np.where(np.random.rand(NP, D) < PRT, 1, 0)
                    step_moves = StepSize * journeys * prt_vectors
                    new_positions = current_pos + step_moves

                    # Reflection for boundary control
                    new_positions = np.where(new_positions < bounds[:, 0], 2 * bounds[:, 0] - new_positions,
                                             new_positions)
                    new_positions = np.where(new_positions > bounds[:, 1], 2 * bounds[:, 1] - new_positions,
                                             new_positions)

                    # Calculate fitness for all new positions
                    new_fitnesses = np.array([func(pos) for pos in new_positions])
                    improved_positions_mask = new_fitnesses < current_fitness

                    # Update only improved positions
                    current_pos[improved_positions_mask] = new_positions[improved_positions_mask]
                    current_fitness = np.min([current_fitness, np.min(new_fitnesses)])

                new_population[i] = current_pos[np.argmin(new_fitnesses)]
                fitness_cache[i] = current_fitness

            pop = new_population

        results.append(np.min(fitness_cache))

    return results


# Example usage:
start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(2)])
results_2D = differential_evolution(parabola, 2, bounds, 2 * 2000)
print("2D Results (rand/1/bin):", results_2D)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(10)])
results_10D = differential_evolution(parabola, 10, bounds, 10 * 2000)
print("10D Results (rand/1/bin):", results_10D)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(30)])
results_30D = differential_evolution(parabola, 30, bounds, 30 * 2000)
print("30D Results (rand/1/bin):", results_30D)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

results_2D_sum = sum(results_2D)
results_10D_sum = sum(results_10D)
results_30D_sum = sum(results_30D)

print(results_2D_sum)
print(results_10D_sum)
print(results_30D_sum)


# Example usage:
start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(2)])
results_2D_best = differential_evolution_best(parabola, 2, bounds, 2 * 2000)
print("2D Results (best/1/bin):", results_2D_best)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(10)])
results_10D_best = differential_evolution_best(parabola, 10, bounds, 10 * 2000)
print("10D Results (best/1/bin):", results_10D_best)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(30)])
results_30D_best = differential_evolution_best(parabola, 30, bounds, 30 * 2000)
print("30D Results (best/1/bin):", results_30D_best)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

results_2D_best_sum = sum(results_2D_best)
results_10D_best_sum = sum(results_10D_best)
results_30D_best_sum = sum(results_30D_best)

print(results_2D_best_sum)
print(results_10D_best_sum)
print(results_30D_best_sum)


# Example usage:
start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(2)])
results_2D_pso = pso(parabola, 2, bounds, 2 * 2000)
print("2D Results (PSO):", results_2D_pso)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(10)])
results_10D_pso = pso(parabola, 10, bounds, 10 * 2000)
print("10D Results (PSO):", results_10D_pso)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(30)])
results_30D_pso = pso(parabola, 30, bounds, 30 * 2000)
print("30D Results (PSO):", results_30D_pso)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

results_2D_pso_sum = sum(results_2D_pso)
results_10D_pso_sum = sum(results_10D_pso)
results_30D_pso_sum = sum(results_30D_pso)

print(results_2D_pso_sum)
print(results_10D_pso_sum)
print(results_30D_pso_sum)

# Example usage:
start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(2)])
results_2D_soma = soma_all_to_one(parabola, 2, bounds, 2 * 2000)
print("2D Results (SOMA All-to-One):", results_2D_soma)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(10)])
results_10D_soma = soma_all_to_one(parabola, 10, bounds, 10 * 2000)
print("10D Results (SOMA All-to-One):", results_10D_soma)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(30)])
results_30D_soma = soma_all_to_one(parabola, 30, bounds, 30 * 2000)
print("30D Results (SOMA All-to-One):", results_30D_soma)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

results_2D_soma_sum = sum(results_2D_soma)
results_10D_soma_sum = sum(results_10D_soma)
results_30D_soma_sum = sum(results_30D_soma)

print(results_2D_soma_sum)
print(results_10D_soma_sum)
print(results_30D_soma_sum)


# Example usage:
start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(2)])
results_2D_soma_all = soma_all_to_all(parabola, 2, bounds, 2 * 2000)
print("2D Results (SOMA All-to-All):", results_2D_soma_all)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(10)])
results_10D_soma_all = soma_all_to_all(parabola, 10, bounds, 10 * 2000)
print("10D Results (SOMA All-to-All):", results_10D_soma_all)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

start_time = time.time()
bounds = np.array([[-100, 100] for _ in range(30)])
results_30D_soma_all = soma_all_to_all(parabola, 30, bounds, 30 * 2000)
print("30D Results (SOMA All-to-All):", results_30D_soma_all)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")

results_2D_soma_all_sum = sum(results_2D_soma_all)
results_10D_soma_all_sum = sum(results_10D_soma_all)
results_30D_soma_all_sum = sum(results_30D_soma_all)

print(results_2D_soma_all_sum)
print(results_10D_soma_all_sum)
print(results_30D_soma_all_sum)




















