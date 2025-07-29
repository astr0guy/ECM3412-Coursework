import numpy as np
import matplotlib.pyplot as plt
import math
import random


# RUNNING INSTRUCTIONS
#
# If needed, install the modules listed above
# - Run the code
# - Gradually, each list of minima will be printed to the console
# - After each problem has run its course, a graph will be printed. This will halt the code until you close it 
# - Once Problem 2 Experiment 3's results have been printed, press any key to exit 


# _ANT CLASS_ 
#
# - Simulates each individual ant

class Ant:

    def __init__(self):
        # Setup for each ant
        self.path = []
        self.fitness = 999999999
        random.seed()

    def roulette_wheel(self, cumulative_path_probabilities):
        # Selection of each probability after normalisation
        # - A random float is selected between 0 and 1
        # - The first probability that is bigger than the generated float is chosen 
        roulette = random.random()
        bin_num = len(cumulative_path_probabilities)
        for j in range(bin_num):
            if roulette <= cumulative_path_probabilities[j]:
                return j

    def pathfind(self, pheromone_graph_matrix):
        item_num = len(pheromone_graph_matrix)
        self.path = []
        for i in range(item_num):
            self.path.append(self.roulette_wheel(self.cumulative_probabilities(np.copy(pheromone_graph_matrix[i]))))

    def fitness_evaluator(self, items, bin_num):
        # Calculates the fitness of the ant's path
        # - Simulates the path of the ant in terms of items in bins
        # - Calculates the fitness by finding the differnce between the most and least filled bins
        #
        # BEAR IN MIND that as this is a minimizer algorithm, lower fitness means a better path
        bins = [0] * bin_num

        for i in range(len(self.path)):
            bins[self.path[i]] += items[i]

        self.fitness = max(bins) - min(bins)

    def cumulative_probabilities(self, pheromone_graph_slice):
        # Nomalises probabilities, then presents them in  cumulative form
        normaliser = 1 / sum(pheromone_graph_slice)
        cumulative_probabilities = np.cumsum(pheromone_graph_slice*normaliser)
        return cumulative_probabilities



# _PHEROMONE GRAPH CLASS_ 
#
# - Simulates the layout of the ladndscape the ants traverse in the form of a streamlined adjacency matrix.


class PheromoneGraph:
    def __init__(graph, item_num, bin_num, evaporation_rate):
        # Initialises the pheromone graph with random floats between 0 and 1
        # Stores the number of items, number of bins and evaporation rate for faster use later
        np.random.seed()
        graph.matrix = np.random.random((item_num, bin_num))
        graph.item_num = item_num
        graph.bin_num = bin_num
        graph.evaporation_rate = evaporation_rate

    def update_pheromones(graph, army_results):
        # Updates the pheromone graph
        # - Adds the pheromones of each ant, proportional to fitness, each path they've taken
        # - Evaporates each pheromone trail in accordance with the evaporation rate
        for i in range(len(army_results)):
            for j in range(len(graph.matrix)):
                graph.matrix[j][army_results[i].path[j]] += 100 / army_results[i].fitness
        graph.matrix *= graph.evaporation_rate





def problem1(army_size, evaporation_rate, iterations):
    # Sets up the hardcoded parameters for problem 1
    bin_num = 10
    fitness_check_limit = 10000
    # Generates the item weights
    items = range(1, 501)
    results = []
    # Runs the traversal algorithm
    for _ in range(iterations):
        results.append((traverse(bin_num, army_size, items, evaporation_rate, fitness_check_limit)))
    return results


def problem2(army_size, evaporation_rate, iterations):
    # Sets up the hardcoded parameters for problem 2
    bin_num = 50
    fitness_check_limit = 10000
    # Generates the item weights
    items = np.array(range(1, 501))
    for i in range(500):
        items[i] = math.ceil(items[i] ** 2 / 2)
    results = []
    # Runs the traversal algorithm
    for _ in range(iterations):
        results.append((traverse(bin_num, army_size, items, evaporation_rate, fitness_check_limit)))
    return results


def traverse(bin_num, army_size, items, evaporation_rate, fitness_check_limit):
    # Sets up the traversal for all ants acording to the parameters of the problem

    # A quality of life variable to streamline memory
    item_num = len(items)

    # Sets up pheromone graph according to parameters
    pheromone_graph = PheromoneGraph(item_num, bin_num, evaporation_rate)

    # Sets up lists to hold ant information
    # - results holds the best ant in each army
    # - army_results holds each army's ants
    results = []
    army_results = []


    # This loop prevents the max number of iterations
    fitness_check_limit = int(fitness_check_limit/army_size)
    for _ in range(fitness_check_limit):
        # This loop separates ants by army
        for i in range(army_size):
            # Create a new Ant, have that ant find a solution, then assess the effectiveness of its solution
            new_ant = Ant()
            new_ant.pathfind(pheromone_graph.matrix)
            new_ant.fitness_evaluator(items, bin_num)
            army_results.append(new_ant)
        # The most recent army deposits their pheromones simultaneously    
        pheromone_graph.update_pheromones(army_results)
        # Saves the army's best ant to results          
        results.append((min([x.fitness for x in army_results])))
        # Clears the army to save memory
        army_results = []
    return results


def graph_output(func, experniment_num):
    #Compiles the results into a graph of 4 subgraphs
    #Each subgraph contains 5 lines representing the minimum of each army's results plotted against the number of armies that had completed traversal in that iteration of the algorithm
    results = func(100, 0.9, 5)
    minima = []
    for a in range(len(results)):
        plt.subplot(2,2,1)
        plt.plot(range(len(results[a])), results[a], linewidth = "0.75")
        plt.title("Army Size 100, Evaporation Rate 0.9")
        plt.xlabel("Army Number")
        plt.ylabel("Fitness")
        minima.append(min(results[a]))
    print("Problem " + str(experniment_num) + " experiment 1 minima:\t" + str(minima))
    results = func(100, 0.6, 5)
    minima = []
    for b in range(len(results)):
        plt.subplot(2,2,2)
        plt.plot(range(len(results[b])), results[b], linewidth = "0.75")
        plt.title("Army Size 100, Evaporation Rate 0.6")
        plt.xlabel("Army Number")
        plt.ylabel("Fitness")
        minima.append(min(results[b]))
    print("Problem " + str(experniment_num) + " experiment 2 minima:\t" + str(minima))
    results = func(10, 0.9, 5)
    minima = []
    for c in range(len(results)):
        plt.subplot(2,2,3)
        plt.plot(range(len(results[c])), results[c], linewidth = "0.75")
        plt.title("Army Size 10, Evaporation Rate 0.9")
        plt.xlabel("Army Number")
        plt.ylabel("Fitness")
        minima.append(min(results[c]))
    print("Problem " + str(experniment_num) + " experiment 3 minima:\t" + str(minima))
    results = func(10, 0.6, 5)
    minima = []
    for d in range(len(results)):
        plt.subplot(2,2,4)
        plt.plot(range(len(results[d])), results[d], linewidth = "0.75")
        plt.title("Army Size 10, Evaporation Rate 0.6")
        plt.xlabel("Army Number")
        plt.ylabel("Fitness")
        minima.append(min(results[d]))
    print("Problem " + str(experniment_num) + " experiment 4 minima:\t" + str(minima))
    plt.suptitle("Problem " + str(experniment_num) + " Results") 
    plt.legend()
    plt.show()


# _MAIN FUNCTION_

# Runs the simulations, graphs the results, and prints the requested minima for each simulation  
graph_output(problem1, 1)
graph_output(problem2, 2)
input()
