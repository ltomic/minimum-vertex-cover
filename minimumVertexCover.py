# run with
# python minimumVertexCover.py ./data/facebook_combined.txt
import random
import time
import matplotlib.pyplot as plt

import sys

def readEdges(input_file):
    edges = set()

    while True:
        line = input_file.readline()

        if line == "":
            break

        line = line.split(" ")

        vertex_a, vertex_b = int(line[0]), int(line[1])

        if vertex_a > vertex_b:
            vertex_a, vertex_b = vertex_b, vertex_a

        edges.add((vertex_a, vertex_b))

    return edges


def readData(input_file):  #uvesti podatke iz .txt
    edges = readEdges(input_file)

    num_vertex = max([max(a, b) for a, b in edges])+1
    W = [1] * num_vertex # TODO: poslije dodati citanje tezina, ako je potrebno

    E = [[] for i in range(num_vertex)]

    for vertex_a, vertex_b in edges:
        E[vertex_a].append(vertex_b)
        E[vertex_b].append(vertex_a)

    return W, E # B, W, DOBRO BI NAM DOSAO INTEGER BROJ VRHOVA (UMJESTO LEN() KASNIJE)


class GeneticAlgorithm:

    def __init__(self, E, W, population_size, n_gen, p_c, p_h, p_m, p_better):
        self.E = E
        self.W = W

        self.n = population_size
        self.n_gen = n_gen

        self.p_c = p_c
        self.p_h = p_h
        self.p_m = p_m
        self.p_better = p_better

    def run(self):
        population = self.generate_initial_population()

        current_best = self.find_best_solution(population)
        WDR_list = [self.W[i] / self.degree(i) for i in range(len(self.E))]
        avg_WDR = sum(WDR_list) / len(self.E)

        gen = 0
        best_by_iteration = []
        while gen < self.n_gen:
            print("generation: ", gen)
            if random.random() < self.p_c:
                p1 = self.binary_tournament_selection(population)
                p2 = self.binary_tournament_selection(population)

                new_solution = self.mutate(self.crossover(p1, p2), WDR_list, avg_WDR)
            else:
                new_solution = self.generate_random_solution()

            new_solution = self.reduction(self.repair(new_solution))

            if new_solution not in population:
                gen += 1

                self.replace_worst_solution(population, new_solution)

                if self.calculate_fitness(new_solution) < self.calculate_fitness(current_best):
                    current_best = new_solution

                best_by_iteration.append(current_best)

        return current_best, best_by_iteration

    def degree(self, x):
        return len(self.E[x])

    def generate_initial_population(self):
        print("Generate initial population")
        population = []

        while len(population) < self.n:
            new_solution = self.generate_solution()

            if new_solution not in population:
                population.append(new_solution)

        return population

    def generate_solution(self):
        start_time = time.time()

        solution = self.reduction(self.repair(self.generate_random_solution()))

        if self.check_vertex_cover(solution) == False:
            print("Generate solution procedure does not give valid vertex cover")
            sys.exit(0)

        end_time = time.time()
        print("Time to generate new solution: ", end_time - start_time)
        return solution

    def generate_random_solution(self): #generira pocetno rjesenje
        return [random.randint(0, 1) if self.degree(i) > 0 else 0 for i in range(len(self.E))]

    def repair(self, solution):
        if random.random() < self.p_h:
            return self.first_repair_heuristic(solution)

        return self.second_repair_heuristic(solution)

    def first_repair_heuristic(self, solution):
        start_time = time.time()

        vertices_not_in_solution = set([i for i in range(len(solution)) if solution[i] == 0])

        edges_not_covered = self.find_edges_not_covered(solution)
        uncovered_edges_cnt = self.calculate_uncovered_edges_for_all_vertices(solution)
        it = 0

        cover_time = 0
        find_vertex_time = 0
        while len(edges_not_covered) > 0:
            it += 1
            start_time_find_vertex = time.time()
            v = self.find_vertex_with_largest_uncovered_edge_weight_ratio(uncovered_edges_cnt,
                    vertices_not_in_solution)
            end_time_find_vertex = time.time()

            find_vertex_time += end_time_find_vertex - start_time_find_vertex

            solution[v] = 1
            vertices_not_in_solution.remove(v)

            start_time_cover = time.time()
            self.cover_edges_not_covered(v, edges_not_covered)
            self.update_uncovered_edges_cnt(uncovered_edges_cnt, solution, v)
            end_time_cover = time.time()

            cover_time += end_time_cover - start_time_cover

        print("Cover time:", cover_time)
        print("Find vertex time:", find_vertex_time)

        if self.check_vertex_cover(solution) == False:
            print("First repair does not return vertex cover")
            sys.exit(0)

        end_time = time.time()

        print("Time first repair: ", end_time - start_time)
        return solution

    def second_repair_heuristic(self, solution):
        start_time = time.time()
        vertices_not_in_solution = set([i for i in range(len(solution)) if solution[i] == 0])

        uncovered_edges_cnt = self.calculate_uncovered_edges_for_all_vertices(solution)

        while self.check_vertex_cover(solution) == False:
            v = random.choice(list(vertices_not_in_solution))
            A = [i for i in E[v] if solution[i] == 0]
            A.append(v)

            s = self.find_vertex_with_largest_uncovered_edge_weight_ratio(uncovered_edges_cnt, A)

            solution[v] = 1
            vertices_not_in_solution.remove(v)
            self.update_uncovered_edges_cnt(uncovered_edges_cnt, solution, v)

        if self.check_vertex_cover(solution) == False:
            print("Second repair does not return vertex cover")
            sys.exit(0)

        end_time = time.time()

        print("Time second repair: ", end_time - start_time)

        return solution

    def weight_to_degree_ratio(self):   #MOZDA ovo racunati u readData
        WDR = [self.W[u] / self.degree(u, self.E) for u in range(len(self.E))]
        averageWDR = sum(self.WDR) / len(self.E)

        return WDR, averageWDR

    # check if vertex is in cover and its edges are covered with neighbor vertices
    def check_covered_vertex(self, x, solution):
        if solution[x] == 0:
            return False

        for i in self.E[x]:
            if solution[i] == 0:
                return False

        return True


    def find_covered_vertices(self, solution):
        return set([i for i in range(len(self.E)) if self.check_covered_vertex(i, solution)])

    def find_vertex_with_highest_weight_degree_ratio(self, covered_vertices, ratios):
        while ratios[-1][0] not in covered_vertices:
            ratios.pop()

        return ratios[-1][0]

    def calculate_weight_degree_ratios(self, vertices):
        return sorted([(u, self.W[u] / self.degree(u)) for u in vertices], key=lambda x: x[1])

    def update_covered_vertices(self, covered_vertices, vertex_to_remove):
        covered_vertices.remove(vertex_to_remove)

        for v in self.E[vertex_to_remove]:
            if v in covered_vertices:
                covered_vertices.remove(v)

    def update_uncovered_edges_cnt(self, uncovered_edges_cnt, solution, v):
        for u in self.E[v]:
            if solution[u] == True:
                continue

            uncovered_edges_cnt[u] -= 1
            uncovered_edges_cnt[v] -= 1

    # izbaciti visak iz rjesenja
    def reduction(self, solution):
        reduction_start_time = time.time()
        num_vertex = len(solution)   #broj vrhova

        # vertex in vertex cover whose edges are covered by neighbor vertex
        find_covered_vertices_start_time = time.time()
        covered_vertices = self.find_covered_vertices(solution)
        find_covered_vertices_end_time = time.time()

        find_covered_vertices_time = find_covered_vertices_end_time -  \
                                    find_covered_vertices_start_time
        vertices_wdr = self.calculate_weight_degree_ratios(covered_vertices)

        # probability with which the vertex having the
        # maximum ratio of weight to degree is selected
        # TODO: namjestiti kao parametar
        p_sc = 0.5
        while len(covered_vertices) > 0:
            if random.random() <= p_sc:
                vertex_to_remove = self.find_vertex_with_highest_weight_degree_ratio(covered_vertices, vertices_wdr)
            else:                       #ovako Singh izbaci random vrh ako ne izbaci najgori
                # TODO: mozda promjeniti covered_vertices u list jer ovo traje O(n)
                vertex_to_remove = random.choice(list(covered_vertices))

            solution[vertex_to_remove] = 0

            find_covered_vertices_start_time = time.time()
            self.update_covered_vertices(covered_vertices, vertex_to_remove)
            find_covered_vertices_end_time = time.time()

            find_covered_vertices_time += find_covered_vertices_end_time -  \
                                        find_covered_vertices_start_time

        if self.check_vertex_cover(solution) == False:
            print("NE VALJA reduction")
            sys.exit(0)

        print("Find covered vertices time:", find_covered_vertices_time)
        reduction_end_time = time.time()

        print("Reduction time:", reduction_end_time - reduction_start_time)
        return solution

    def calculate_uncovered_edges_cnt(self, solution, x):
        uncovered_edges = 0

        for v in self.E[x]:
            if solution[v] == 0:
                uncovered_edges += 1

        return uncovered_edges

    def calculate_uncovered_edges_for_all_vertices(self, solution):
        uncovered_edges_cnt = [0 for i in range(len(solution))]

        for u in range(len(solution)):
            if solution[u] == 1:
                continue

            for v in self.E[u]:
                if solution[v] == 0:
                    uncovered_edges_cnt[u] += 1

        return uncovered_edges_cnt

    def find_vertex_with_largest_uncovered_edge_weight_ratio(self, uncovered_edges_cnt, vertices):
        sol = -1
        max_ratio = -1

        for i in vertices:
            ratio = uncovered_edges_cnt[i] / self.W[i]

            if ratio > max_ratio: # TODO: jos dodati tu random
                max_ratio = ratio
                sol = i

        return sol

    def check_vertex_cover(self, solution):
        for i in range(len(solution)):
            if solution[i] == 0:
                for j in self.E[i]:
                    if solution[j] == 0:
                        return False

        return True

    def find_edges_not_covered(self, solution):
        edges_not_covered = set()

        for i in range(len(solution)):
            if solution[i] == 1:
                continue

            for b in self.E[i]:
                if solution[b] == False:
                    edges_not_covered.add(frozenset([i, b]))

        return edges_not_covered

    def cover_edges_not_covered(self, u, edges_not_covered):
        for v in self.E[u]:
            if frozenset([u, v]) in edges_not_covered:
                edges_not_covered.remove(frozenset([u, v]))

    def calculate_fitness(self, solution):
        return sum([self.W[i] for i in range(len(solution)) if solution[i] == 1])

    def find_best_solution(self, population):
        best_solution = population[0]
        best_score = self.calculate_fitness(best_solution)

        for solution in population:
            current_score = self.calculate_fitness(solution)

            if current_score < best_score:
                best_solution = solution
                best_score = current_score

        return best_solution

    def find_worst_solution_index(self, population):
        worst_solution_index = 0
        worst_score = self.calculate_fitness(population[worst_solution_index])

        for i in range(len(population)):
            current_score = self.calculate_fitness(population[i])

            if current_score > worst_score:
                worst_solution_index = i
                worst_score = current_score

        return worst_solution_index

    def replace_worst_solution(self, population, new_solution):
        worst_solution_index = self.find_worst_solution_index(population)

        population[worst_solution_index] = new_solution

    def binary_tournament_selection(self, population):
        a, b = random.sample(population, k = 2)

        fitness_a, fitness_b = self.calculate_fitness(a), self.calculate_fitness(b)

        if random.random() > self.p_better:
            return b if fitness_a > fitness_b else a

        return a if fitness_a > fitness_b else b

    def mutate(self, solution, WDR_list, average_WDR):
        for i in range(len(solution)):
            if solution[i] == 1:
                if random.random() < self.p_m:
                    solution[i] = 0
            else:
                if random.random() < self.p_m and WDR_list[i] < average_WDR:  #redoslijed oko and?
                    solution[i] = 1

        return solution

    def crossover(self, parent_1, parent_2): # TODO: jos dodati random
        child = parent_2[:]
        p_1 = self.calculate_fitness(parent_1)
        p_2 = self.calculate_fitness(parent_2)
        p1 = p_2 / (p_1 + p_2)

        for i in range(len(parent_1)):
            if random.random() < p1:
                child[i] = parent_1[i]

        return child

def plot_results_by_iteration(best_by_iteration):
    fig, ax = plt.subplots()

    ax.plot(range(len(best_by_iteration)), best_by_iteration, linewidth = 2.0)

    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename, 'r') as input_file:
        W, E = readData(input_file)

    population_size = 100
    n_gen = 100
    algorithm = GeneticAlgorithm(E, W, population_size, n_gen,
            p_c = 0.9, p_h = 0.2, p_m = 0.05, p_better = 0.8)

    solution, best_by_iteration = algorithm.run()

    print(algorithm.calculate_fitness(solution))
    print(algorithm.check_vertex_cover(solution))

    plot_results_by_iteration([algorithm.calculate_fitness(solution) for solution \
            in best_by_iteration])

