import multiprocessing
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


def rastrigin(v):
    x = np.linalg.norm(v)
    n = len(v)
    total = 10 * n
    for i in range(n):
        total += x ** 2 - 10 * np.cos(2 * np.pi * x)
    return total


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


def ackley(v):
    x = np.linalg.norm(v)
    n = len(v)
    sum1 = sum([xi ** 2 for xi in v])
    sum2 = sum([np.cos(2 * np.pi * xi) for xi in v])
    return -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e


def reuleaux_triangle(v):
    x = np.linalg.norm(v)
    return np.maximum.reduce([np.abs(x), np.abs(x - 2), np.abs(x + 2)]) - 1


def dirichlet(v):
    x = np.linalg.norm(v)
    return np.sin(x) / np.where(x == 0, 1, x)  # Avoid division by zero


def michalewicz(v):
    x = np.linalg.norm(v)
    n = len(v)
    return -sum([np.sin(xi) * (np.sin((i + 1) * xi ** 2 / np.pi)) ** (2 * 10) for i, xi in enumerate(v)])


def logarithmic_decay(v):
    x = np.linalg.norm(v)
    return -np.log(x + 1)


def inverted_sine(v):
    x = np.linalg.norm(v)
    return -np.sin(x)


def hyperbolic_cosine(v):
    x = np.linalg.norm(v)
    return np.cosh(x)


######################
#####################

# NEW FUNCTIONS
def whitley(v):
    return np.sum([np.sum([(100 * ((v[i] ** 2 - v[j]) ** 2) + (1 - v[i]) ** 2) ** 2 / 4000 - np.cos(
        100 * ((v[i] ** 2 - v[j]) ** 2) + (1 - v[i]) ** 2) + 1 for j in range(len(v))]) for i in range(len(v))])


def griewank(v):
    part1 = np.sum(v ** 2 / 4000.0)
    part2 = np.prod(np.cos(v / np.sqrt(np.arange(1, len(v) + 1))))
    return 1 + part1 - part2


def levy(v):
    w = 1 + (v - 1) / 4
    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
    return term1 + term2 + term3


def eggholder(v):
    if len(v) < 2:
        raise ValueError("Eggholderova funkce vyžaduje minimálně 2 dimenze.")

    summands = [
        -(v[i + 1] + 47) * np.sin(np.sqrt(np.abs(v[i + 1] + v[i] / 2 + 47))) - v[i] * np.sin(
            np.sqrt(np.abs(v[i] - (v[i + 1] + 47))))
        for i in range(len(v) - 1)
    ]
    return np.sum(summands)


def rosenbrock(v):
    return np.sum(100.0 * (v[1:] - v[:-1] ** 2) ** 2 + (v[:-1] - 1) ** 2)


def zakharov(v):
    sum1 = np.sum(v ** 2)
    sum2 = np.sum(0.5 * np.arange(1, len(v) + 1) * v) ** 2
    sum3 = np.sum(0.5 * np.arange(1, len(v) + 1) * v) ** 4
    return sum1 + sum2 + sum3


def salomon(v):
    norm_v = np.linalg.norm(v)
    return 1 - np.cos(2 * np.pi * norm_v) + 0.1 * norm_v


def fletcher_powell(v):
    d = len(v)
    a = np.array([[np.random.uniform(-100, 100) for _ in range(d)] for _ in range(d)])
    b = np.array([np.random.uniform(-np.pi, np.pi) for _ in range(d)])
    c = np.array([np.random.uniform(-100, 100) for _ in range(d)])
    sum_terms = np.sum([
        (a[i, :] @ v + b[i] - c[i]) ** 2
        for i in range(d)
    ])
    return sum_terms


def schwefel(v):
    return 418.9829 * len(v) - np.sum(v * np.sin(np.sqrt(np.abs(v))))


def dixon_price(v):
    term1 = (v[0] - 1) ** 2
    term2 = np.sum([(i + 1) * (2 * v[i] ** 2 - v[i - 1]) ** 2 for i in range(1, len(v))])
    return term1 + term2


functions_to_finish = [whitley, griewank, levy, eggholder, rosenbrock, zakharov, salomon, fletcher_powell, schwefel,
                       dixon_price]


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
    plt.savefig(f"./Plots_New/{filename2D}.png")
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
    plt.savefig(f"./Plots_New/{filename3D}.png")
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
    rastrigin,
    quadratic_growth,
    quadratic_decay,
    square_root,
    cubic,
    logarithmic,
    ackley,
    reuleaux_triangle,
    dirichlet,
    michalewicz,
    logarithmic_decay,
    inverted_sine,
]


# Create 2D & 3D plots for all the functions
# for test_function in functions_to_finish:
#    plot_function(test_function)


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

        for _ in range(int(FEs / NP)):
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

        for _ in range(int(FEs / NP)):
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
                new_positions[i] = np.where(new_positions[i] < bounds[:, 0], 2 * bounds[:, 0] - new_positions[i],
                                            new_positions[i])
                new_positions[i] = np.where(new_positions[i] > bounds[:, 1], 2 * bounds[:, 1] - new_positions[i],
                                            new_positions[i])

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

        for _ in range(int(FEs / NP)):
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


def single_repetition_soma_all_to_one(func, D, bounds, FEs, NP, PathLength, StepSize, PRT):
    # Initialize population
    pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    new_population = np.copy(pop)
    fitness_cache = np.array([func(ind) for ind in pop])

    for _ in range(int(FEs / NP)):
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

    return fitness_cache[np.argmin(fitness_cache)]


def parallel_soma_all_to_one(func, D, bounds, FEs, repetitions=30, PathLength=3, StepSize=0.11, PRT=0.7):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    # Create a pool of worker processes
    num_cores = 10  # Or set it to a desired number of cores or to multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Use pool.map to parallelize the repetitions
    args = [(func, D, bounds, FEs, NP, PathLength, StepSize, PRT) for _ in range(repetitions)]
    results = pool.starmap(single_repetition_soma_all_to_one, args)

    pool.close()
    pool.join()

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
                # Broadcast difference between individual i and all others
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


def single_repetition_soma_all_to_all(func, D, bounds, FEs, NP, PathLength, StepSize, PRT):
    pop = np.random.rand(NP, D) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    fitness_cache = np.array([func(ind) for ind in pop])

    for _ in range(int(FEs / NP)):
        print(_)
        new_population = pop.copy()

        # For each individual in the population
        for i in range(NP):
            # Broadcast difference between individual i and all others
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

    return np.min(fitness_cache)


def parallel_soma_all_to_all(func, D, bounds, FEs, repetitions=30, PathLength=3, StepSize=0.11, PRT=0.7):
    # Define population size based on dimension
    if D == 2:
        NP = 10
    elif D == 10:
        NP = 20
    elif D == 30:
        NP = 50
    else:
        raise ValueError("Unsupported dimension!")

    # Create a pool of worker processes
    num_cores = 10  # Or set it to a desired number of cores or to multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_cores)

    # Use pool.map to parallelize the repetitions
    args = [(func, D, bounds, FEs, NP, PathLength, StepSize, PRT) for _ in range(repetitions)]
    results = pool.starmap(single_repetition_soma_all_to_all, args)

    pool.close()
    pool.join()

    return results


def calculate_list_AVG(list_to_be_summed):
    return np.mean(list_to_be_summed)


def print_results(func_name, dimensions, results, algorithm_name, AVG):
    # return f"AVG for {func_name.upper()} with {algorithm_name.upper()} {dimensions} algorithm:", f"{AVG}"
    return f"Results for {func_name.upper()} with {algorithm_name.upper()} {dimensions} algorithm:", results, f"AVG = {AVG}"


def write_to_file(file_name, data_tuple):
    # Convert tuple to string
    data_str = '\n'.join(map(str, data_tuple))
    with open(file_name, "a") as file:
        # Append the new data to the file
        file.write(data_str)
        file.write('\n')  # add a newline for separation


bounds_2D = np.array([[-100, 100] for _ in range(2)])
bounds_10D = np.array([[-100, 100] for _ in range(10)])
bounds_30D = np.array([[-100, 100] for _ in range(30)])


def run_algo_over_functions_2D(algo):
    iteration = 0
    for func in functions_to_finish:
        start_time = time.time()
        result = algo(func, 2, bounds_2D, 2 * 2000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg = calculate_list_AVG(result)
        data = print_results(func.__name__, "2D", result, algo.__name__, avg)
        iteration += 1
        print(iteration)
        print(f"Elapsed Time: {elapsed_time} seconds")
        write_to_file(f"{algo.__name__}_2D_NEW.txt", data)


def run_algo_over_functions_10D(algo):
    iteration = 0
    for func in functions_to_finish:
        start_time = time.time()
        result = algo(func, 10, bounds_10D, 10 * 2000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg = calculate_list_AVG(result)
        data = print_results(func.__name__, "10D", result, algo.__name__, avg)
        iteration += 1
        print(iteration)
        print(f"Elapsed Time: {elapsed_time} seconds")
        write_to_file(f"{algo.__name__}_10D_NEW.txt", data)


def run_algo_over_functions_30D(algo):
    iteration = 0
    for func in functions_to_finish:
        start_time = time.time()
        result = algo(func, 30, bounds_30D, 30 * 2000)
        end_time = time.time()
        elapsed_time = end_time - start_time
        avg = calculate_list_AVG(result)
        data = print_results(func.__name__, "30D", result, algo.__name__, avg)
        iteration += 1
        print(iteration)
        print(f"Elapsed Time: {elapsed_time} seconds")
        write_to_file(f"{algo.__name__}_30D_NEW.txt", data)


def worker(args):
    algo, dimension_function = args
    dimension_function(algo)


if __name__ == '__main__':
    algorithms = [differential_evolution, differential_evolution_best, pso]
    dimension_functions = [run_algo_over_functions_2D, run_algo_over_functions_10D, run_algo_over_functions_30D]

    # Create a flattened list of all combinations of algorithms and dimension functions
    tasks = [(algo, dim_func) for algo in algorithms for dim_func in dimension_functions]

    # Use all available cores
    num_cores = multiprocessing.cpu_count()

    # Create a process pool and start executing
    with multiprocessing.Pool(num_cores) as pool:
        pool.map(worker, tasks)

    print("All tasks finished!")

    """
    start_time = time.time()
    run_algo_over_functions_2D(parallel_soma_all_to_one)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    start_time = time.time()
    run_algo_over_functions_10D(parallel_soma_all_to_one)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    start_time = time.time()
    run_algo_over_functions_30D(parallel_soma_all_to_one)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    ##################

    start_time = time.time()
    run_algo_over_functions_2D(parallel_soma_all_to_all)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    start_time = time.time()
    run_algo_over_functions_10D(parallel_soma_all_to_all)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")

    start_time = time.time()
    run_algo_over_functions_30D(parallel_soma_all_to_all)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time} seconds")
    """
