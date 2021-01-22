import math

import numpy as np

from chromosome import Chromosome
from population import Population


class Solver:
    def __init__(self, Z, distances, P, ng, Pc, Pm, Pmb, k=None):

        self.distances = distances

        # Find k number based on number of customers K and
        # acceptable TSPTW subproblem size Z
        if k is None:
            K = len(distances)
            self.k = math.ceil(K / Z)
        else:
            self.k = k
        self.P = P
        self.ng = 10

        self.Pc = Pc
        self.Pm = Pm
        self.Pmb = Pmb

        self.current_population = Population(self.P, self.k, self.distances)
        self.best_chromosome = Chromosome(self.k, self.distances)

    def make_cluster_from_medoids(self):
        medoids = self.best_chromosome.genes
        points_size = int(np.ceil(self.distances[0].size / medoids.size))
        result = np.array([[-1 for _ in range(points_size)] for _ in range(medoids.size)])

        deprecated = np.copy(medoids)

        for i, gene in enumerate(medoids):
            for j in range(points_size):
                if j == 0:
                    result[i][0] = gene
                    continue

                cur_min_ind = -1
                cur_min = math.inf
                for k, el in enumerate(self.distances[gene]):
                    if gene != k and el < cur_min and k not in deprecated:
                        cur_min = el
                        cur_min_ind = k
                result[i][j] = cur_min_ind
                deprecated = np.append(deprecated, cur_min_ind)

        return result

    def solve(self):
        self.current_population.generate_random_population()
        self.current_population.calculate_fitness()
        self.save_new_best_chromosome(self.current_population)

        print("---------")
        print("Iteration {}".format(0))
        print("Best chromosome fitness {}".format(self.best_chromosome.fitness))
        print("All chromosomes genes: {}".format(
            [chromosome.genes.tolist() for chromosome in self.current_population.chromosomes]))
        print("Best genes {}".format(self.best_chromosome.genes))
        print("---------")

        for i in range(1, self.ng):
            self.current_population = self.current_population.selection()
            self.current_population.dmx_crossover(self.Pc, self.Pmb)
            self.current_population.mutate(self.Pm)

            self.current_population.calculate_fitness()
            self.save_new_best_chromosome(self.current_population)

            print("---------")
            print("Iteration: {}".format(i))
            print("Best chromosome fitness: {}".format(self.best_chromosome.fitness))
            print("All chromosomes genes: {}".format(
                [chromosome.genes.tolist() for chromosome in self.current_population.chromosomes]))
            print("Best genes: {}".format(self.best_chromosome.genes))
            print("---------")

        return self.make_cluster_from_medoids()

    def save_new_best_chromosome(self, population):
        chrom_fitness = population.find_best_chromosome().fitness
        print("cur fitness: {}".format(chrom_fitness))
        if chrom_fitness <= self.best_chromosome.fitness:
            self.best_chromosome = population.find_best_chromosome()
