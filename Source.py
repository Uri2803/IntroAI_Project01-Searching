import numpy as np
import random
from typing import List
import heapq
import sys
from queue import PriorityQueue
def menu():
    print("# 1: ")
    print("# 2: Branch và Bound Algorithms")
    print("# 3: Local Beam Search Algorithms")
    print("# 4: Genetic Algorithms")
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
        f.write(' '.join(str(x) for x in knapsack))
#---------------------------------

def Branch_and_Bound_Algorithm():
    # class Item:
    #     def __init__(self, weight: float, value: int, class_label: int):
    #         self.weight = weight
    #         self.value = value
    #         self.class_label = class_label
    #         self.ratio = value / weight
    class knapsack_bnb:
        def __init__(self, W: int, m: int, w: 'list[int]', v: 'list[int]', c: 'list[int]') -> None:
                self.W = W 
                self.m = m 
                self.w = w 
                self.v = v 
                self.c = c 
                self.n = len(w) 
                self.best_value = 0 
                self.best_items = []
        def bound(self, k: int, weight: float, value: int, taken: List[bool]) -> float:
            # If weight exceeds the limit of the knapsack, return 0
            if weight >= self.W:
                return 0
            # Calculate the upper bound value
            bound = value
            # Iterate through the remaining items and calculate the upper bound
            for i in range(k, self.n):
                if not taken[i] and weight + self.w[i] <= self.W:
                    bound += self.v[i]
                    weight += self.w[i]
                else:
                    # The remaining part of the item cannot be selected, 
                    # we will take a part of it and calculate the upper bound value
                    remaining_weight = self.W - weight
                    bound += remaining_weight * (self.v[i] / self.w[i])
                    break
            
            return bound
        def knapsack(self, k: int, weight: int, value: int, taken: 'list[bool]'):
                if weight <= self.W and value > self.best_value: # Check if the current weight is less than or equal to the weight of the knapsack and 
                                                        # the current value is greater than the best value, then Update 
                    self.best_value = value
                    self.best_items = taken[:]
                if k == self.n: # Check if all items have been surveyed => Return
                    return
                # If not all items have been surveyed, create a descending sorted list and survey each item
                sorted_items = sorted([(i, self.v[i]/self.w[i]) for i in range(k, self.n)], key=lambda x: -x[1])
                for i, _ in sorted_items:
                    if weight + self.w[i] <= self.W: # If the item can be added to the knapsack without overloading the weight, recursively call the function with this item added 
                                                    # to the knapsack and check the conditions as above
                        taken[i] = True
                        self.knapsack(i + 1, weight + self.w[i], value + self.v[i], taken)
                        taken[i] = False
                    if self.bound(i + 1, weight, value, taken) > self.best_value:
                        taken[i] = False
                        self.knapsack(i + 1, weight, value, taken)
        def solve(self) -> 'tuple[int, list[int]]':
                taken = [False] * self.n
                self.knapsack(0, 0, 0, taken)
              
                return str(self.best_value), ''.join([str(int(i)) for i in self.best_items])


    input_file = "INPUT.txt"
    W, m, w, v, c = read_input_file(input_file)
    bb=knapsack_bnb(W, m, w, v, c)
    max_value, knapsack =  bb.solve()
    output_file = "OUTPUT.txt"
    write_output_file(output_file, max_value, knapsack)

#-----------------------------
def  Local_Beam_Search_Algorithms():
    def local_beam_search(W, m, w, v, c, k=15):
        # Get the number of items
        n = len(w)

        # Define fitness function
        def fitness(solution):
            # Calculate the total weight of the solution
            weight = np.sum(solution * w)
            # Check if the weight exceeds the knapsack's capacity
            if weight > W:
                return 0  # infeasible solution
            else:
                # Calculate the total value of the solution
                value = np.sum(solution * v)
                return value

        # Define generate function
        def generate(solution):
            # Make a copy of the solution to avoid changing the original solution
            new_solution = solution.copy()
            # Choose a random item to flip its value (either 0 or 1)
            i = np.random.randint(n)
            new_solution[i] = 1 - new_solution[i]
            return new_solution

        # Initialize with k random solutions
        solutions = [np.random.randint(2, size=n) for _ in range(k)]
        fitnesses = [fitness(s) for s in solutions]

        while True:
            # Generate k new candidate solutions for each current solution
            new_solutions = []
            new_fitnesses = []
            for i in range(k):
                for _ in range(k):
                    # Generate a new solution by flipping a random item's value
                    new_solution = generate(solutions[i])
                    # Calculate the fitness of the new solution
                    new_fitness = fitness(new_solution)
                    # Add the new solution and its fitness to the lists
                    new_solutions.append(new_solution)
                    new_fitnesses.append(new_fitness)

            # Select the k best solutions
            best_indices = np.argsort(new_fitnesses)[::-1][:k]
            solutions = [new_solutions[i] for i in best_indices]
            fitnesses = [new_fitnesses[i] for i in best_indices]

            # Check if stopping criterion is met
            if np.std(fitnesses) < 1e-6:
                break

        # Choose the best solution found
        best_index = np.argmax(fitnesses)
        best_solution = solutions[best_index]
        best_fitness = fitnesses[best_index]
        # Return the best solution found
        return best_fitness, best_solution
    input_file = "INPUT.txt"
    W, m, w, v, c = read_input_file(input_file)
    max_value, knapsack = local_beam_search(W, m, w, v, c, k=35)
    # Write output file
    output_file = "OUTPUT.txt"
    write_output_file(output_file, max_value, knapsack)
    
#---------------------------------
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
    if select==2:
        Branch_and_Bound_Algorithm()
    if select==3:
         Local_Beam_Search_Algorithms()
    if select==4:
        Genetic_Algorithms()
    if select > 4 and select == 0:
        print(" No select - end! ")
