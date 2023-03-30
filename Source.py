import numpy as np
import random
def menu():
    print("# 1: ")
    print("# 2: ")
    print("# 3: ")
    print("# 4: Genetic_Algorithms")
# Read input file
def read_input_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        W = int(lines[0].strip()) # Knapsack's storage capacity
        m = int(lines[1].strip()) # number of classes
        w = np.array([float(x) for x in lines[2].strip().split(',')])
        v = np.array([int(x) for x in lines[3].strip().split(',')])
        c = np.array([int(x) for x in lines[4].strip().split(',')])
    return W, m, w, v, c

# Write output file
def write_output_file(file_name, max_value, knapsack):
    with open(file_name, 'w') as f:
        f.write(str(max_value) + '\n')
        f.write(' ,'.join(str(x) for x in knapsack))

# Define fitness function


def Genetic_Algorithms():

    def fitness_function(chromosome, W, w, v):
        weight = np.sum(chromosome * w)
        value = np.sum(chromosome * v)
        if weight > W:
            return 0
        else:
            return value

# Define selection operator
    def selection(population, fitness_values, num_parents):
        # Tìm vị trí của những thể cá nhân có giá trị fitness tốt nhất
        best_fitness_indices = np.argsort(fitness_values)[-num_parents:]
        
        # Lấy các thể cá nhân có giá trị fitness tốt nhất từ quần thể
        parents = [population[i] for i in best_fitness_indices]
        
        return parents

# Define crossover operator
    def crossover(parents, num_offsprings):
        offsprings = []
        for i in range(num_offsprings):
            parent1 = parents[random.randint(0, len(parents)-1)]
            parent2 = parents[random.randint(0, len(parents)-1)]
            crossover_point = random.randint(1, len(parent1)-1)
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offsprings.append(offspring)
        return offsprings

# Define mutation operator
    def mutation(offsprings, mutation_rate):
        for offspring in offsprings:
            for i in range(len(offspring)):
                if random.random() < mutation_rate:
                    offspring[i] = 1 - offspring[i]
        return offsprings

    def generate_random(length):
        return np.random.randint(2, size=length)

# Define generate_population function
    def generate_population(population_size, length):
        return [generate_random(length) for _ in range(population_size)]

# Define genetic algorithm function
    def genetic_algorithm(W, m, w, v, c, num_generations, population_size, num_parents, num_offsprings, mutation_rate):
        length = len(w)
        population = generate_population(population_size, length)
        best_max_value = -1
        for i in range(num_generations):
            fitness_values = np.array([fitness_function(chromosome, W, w, v) for chromosome in population])
            parents = selection(population, fitness_values, num_parents)
            offsprings = crossover(parents, num_offsprings)
            offsprings = mutation(offsprings, mutation_rate)
            population = parents + offsprings
            fitness_values = np.array([fitness_function(chromosome, W, w, v) for chromosome in population])
            max_value = np.max(fitness_values)
            if max_value > best_max_value:
                best_max_value = max_value
                best_chromosome = population[np.argmax(fitness_values)]
        knapsack = [1 if gene == 1 else 0 for gene in best_chromosome]
        return best_max_value, knapsack

    # Set random seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    # Set parameters
    num_generations = 500
    population_size = 200
    num_parents = 100
    num_offsprings = population_size - num_parents
    mutation_rate = 0.1
     # Read input file
    input_file = "INPUT.txt"
    W, m, w, v, c = read_input_file(input_file)
    max_value, knapsack = genetic_algorithm(W, m, w, v, c, num_generations, population_size, num_parents, num_offsprings, mutation_rate)
    # Write output file
    output_file = "OUTPUT.txt"
    write_output_file(output_file, max_value, knapsack)

if __name__ == '__main__':
    menu()
    select =int (input("Enter selection algorithm: "))
  
    # if select==1: 
    #    #call function
    # if select==2:
    #    #call function
    # if select==3:
    #     #call function
    if select==4:
        Genetic_Algorithms()
    if select > 4 and select == 0:
        print(" No select - end! ")

