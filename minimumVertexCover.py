# run with
# python minimumVertexCover.py ./data/facebook_combined.txt
import random
import time

import sys

def degree(x, E):
    return len(E[x])

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

def weight_to_degree_ratio(E, W):   #MOZDA ovo racunati u readData
    WDR = [W[u] / degree(u, E) for u in range(len(E))]
    averageWDR = sum(WDR) / len(E)

    return WDR, averageWDR

# check if vertex is in cover and its edges are covered with neighbor vertices
def check_covered_vertex(x, state, E):
    if state[x] == 0:
        return False

    for i in E[x]:
        if state[i] == 0:
            return False

    return True


def find_covered_vertices(state, E):
    return set([i for i in range(len(E)) if check_covered_vertex(i, state, E)])

def find_vertex_with_highest_weight_degree_ratio(covered_vertices, ratios):
    while ratios[-1][0] not in covered_vertices:
        ratios.pop()

    return ratios[-1][0]

def calculate_weight_degree_ratios(vertices, W, E):
    return sorted([(u, W[u] / degree(u, E)) for u in vertices], key=lambda x: x[1])

def update_covered_vertices(covered_vertices, E, vertex_to_remove):
    covered_vertices.remove(vertex_to_remove)

    for v in E[vertex_to_remove]:
        if v in covered_vertices:
            covered_vertices.remove(v)


# izbaciti visak iz rjesenja
def reduction(state, E, W):
    reduction_start_time = time.time()
    num_vertex = len(state)   #broj vrhova

    # vertex in vertex cover whose edges are covered by neighbor vertex
    find_covered_vertices_start_time = time.time()
    covered_vertices = find_covered_vertices(state, E)
    find_covered_vertices_end_time = time.time()

    find_covered_vertices_time = find_covered_vertices_end_time -  \
                                find_covered_vertices_start_time
    vertices_wdr = calculate_weight_degree_ratios(covered_vertices, W, E)

    # probability with which the vertex having the
    # maximum ratio of weight to degree is selected
    # TODO: namjestiti kao parametar
    p_sc = 0.5
    while len(covered_vertices) > 0:
        if random.random() <= p_sc:
            vertex_to_remove = find_vertex_with_highest_weight_degree_ratio(covered_vertices, vertices_wdr)
        else:                       #ovako Singh izbaci random vrh ako ne izbaci najgori
            # TODO: mozda promjeniti covered_vertices u list jer ovo traje O(n)
            vertex_to_remove = random.choice(list(covered_vertices))

        state[vertex_to_remove] = 0

        find_covered_vertices_start_time = time.time()
        update_covered_vertices(covered_vertices, E, vertex_to_remove)
        find_covered_vertices_end_time = time.time()

        find_covered_vertices_time += find_covered_vertices_end_time -  \
                                    find_covered_vertices_start_time

    if check_vertex_cover(state, E) == False:
        print("NE VALJA reduction")
        sys.exit(0)

    print("Find covered vertices time:", find_covered_vertices_time)
    reduction_end_time = time.time()

    print("Reduction time:", reduction_end_time - reduction_start_time)
    return state

def calculate_uncovered_edges(state, E, x):
    uncovered_edges = 0

    for v in E[x]:
        if state[v] == 0:
            uncovered_edges += 1

    return uncovered_edges

def calculate_uncovered_edges_for_all_vertices(state, E):
    uncovered_edges_cnt = [0 for i in range(len(state))]

    for u in range(len(state)):
        if state[u] == 1:
            continue

        for v in E[u]:
            if state[v] == 0:
                uncovered_edges_cnt[u] += 1

    return uncovered_edges_cnt

def find_vertex_with_largest_uncovered_edge_weight_ratio(uncovered_edges_cnt,
        candidate_vertices, W):
    sol = -1
    max_ratio = -1

    for i in candidate_vertices:
        ratio = uncovered_edges_cnt[i] / W[i]

        if ratio > max_ratio: # TODO: jos dodati tu random
            max_ratio = ratio
            sol = i

    return sol

def check_vertex_cover(state, E):
    for i in range(len(state)):
        if state[i] == 0:
            for j in E[i]:
                if state[j] == 0:
                    return False

    return True

def find_edges_not_covered(state, E):
    edges_not_covered = set()

    for i in range(len(state)):
        if state[i] == 1:
            continue

        for b in E[i]:
            if state[b] == False:
                edges_not_covered.add(frozenset([i, b]))

    return edges_not_covered

def cover_edges_not_covered(u, edges_not_covered, E):
    for v in E[u]:
        if frozenset([u, v]) in edges_not_covered:
            edges_not_covered.remove(frozenset([u, v]))

def update_uncovered_edges_cnt(uncovered_edges_cnt, state, v, E):
    for u in E[v]:
        if state[u] == True:
            continue

        uncovered_edges_cnt[u] -= 1
        uncovered_edges_cnt[v] -= 1

def first_repair_heuristic(state, E, W):
    start_time = time.time()

    vertices_not_in_solution = set([i for i in range(len(state)) if state[i] == 0])

    edges_not_covered = find_edges_not_covered(state, E)
    uncovered_edges_cnt = calculate_uncovered_edges_for_all_vertices(state, E)
    it = 0

    cover_time = 0
    find_vertex_time = 0
    while len(edges_not_covered) > 0:
        it += 1
        start_time_find_vertex = time.time()
        v = find_vertex_with_largest_uncovered_edge_weight_ratio(uncovered_edges_cnt,
                vertices_not_in_solution, W)
        end_time_find_vertex = time.time()

        find_vertex_time += end_time_find_vertex - start_time_find_vertex

        state[v] = 1
        vertices_not_in_solution.remove(v)

        start_time_cover = time.time()
        cover_edges_not_covered(v, edges_not_covered, E)
        update_uncovered_edges_cnt(uncovered_edges_cnt, state, v, E)
        end_time_cover = time.time()

        cover_time += end_time_cover - start_time_cover

    print("Cover time:", cover_time)
    print("Find vertex time:", find_vertex_time)

    if check_vertex_cover(state, E) == False:
        print("First repair does not return vertex cover")
        sys.exit(0)

    end_time = time.time()

    print("Time first repair: ", end_time - start_time)
    return state

def second_repair_heuristic(state, E, W):
    start_time = time.time()
    vertices_not_in_solution = set([i for i in range(len(state)) if state[i] == 0])

    uncovered_edges_cnt = calculate_uncovered_edges_for_all_vertices(state, E)

    while check_vertex_cover(state, E) == False:
        v = random.choice(list(vertices_not_in_solution))
        A = [i for i in E[v] if state[i] == 0]
        A.append(v)

        s = find_vertex_with_largest_uncovered_edge_weight_ratio(uncovered_edges_cnt,
                A, W)

        state[v] = 1
        vertices_not_in_solution.remove(v)
        update_uncovered_edges_cnt(uncovered_edges_cnt, state, v, E)

    if check_vertex_cover(state, E) == False:
        print("NE VALJA second repair")
        sys.exit(0)

    end_time = time.time()

    print("Vrijeme second repair: ", end_time - start_time)

    return state

def repair(state, E, W, p_h = 0.2):
    if random.random() < p_h:
        return first_repair_heuristic(state, E, W)

    return second_repair_heuristic(state, E, W)

def generate_random_solution(num_vertex): #generira pocetno rjesenje
    return [random.randint(0, 1) if degree(i, E) > 0 else 0 for i in range(num_vertex)]

def generate_solution(E, W):
    start_time = time.time()

    solution = reduction(repair(generate_random_solution(len(E)), E, W), E, W)

    if check_vertex_cover(solution, E) == False:
        print("Generate solution procedure does not give valid vertex cover")
        sys.exit(0)

    end_time = time.time()
    print("Time to generate new solution: ", end_time - start_time)
    return solution

def generate_initial_population(n, E, W):
    print("Generate initial population")
    population = []

    while len(population) < n:
        new_solution = generate_solution(E, W)

        if new_solution not in population:
            population.append(new_solution)

    return population

def calculate_fitness(solution, W):
    return sum([W[i] for i in range(len(solution)) if solution[i] == 1])

def find_best_solution(population, W):
    best_solution = population[0]
    best_score = calculate_fitness(best_solution, W)

    for solution in population:
        current_score = calculate_fitness(solution, W)

        if current_score < best_score:
            best_solution = solution
            best_score = current_score

    return best_solution

def find_worst_solution_index(population, W):
    worst_solution_index = 0
    worst_score = calculate_fitness(population[worst_solution_index], W)

    for i in range(len(population)):
        current_score = calculate_fitness(population[i], W)

        if current_score > worst_score:
            worst_solution_index = i
            worst_score = current_score

    return worst_solution_index

def replace_worst_solution(population, new_solution, W):
    worst_solution_index = find_worst_solution_index(population, W)

    population[worst_solution_index] = new_solution

def binary_tournament_selection(population, W, p_better = 0.8):
    a, b = random.sample(population, k = 2)

    fitness_a, fitness_b = calculate_fitness(a, W), calculate_fitness(b, W)

    if random.random() > p_better:
        return b if fitness_a > fitness_b else a

    return a if fitness_a > fitness_b else b

def mutate(solution, WDR_list, average_WDR, p_m = 0.05): #TODO: p_m=???
    for i in range(len(solution)):
        if solution[i] == 1:
            if random.random() < p_m:
                solution[i] = 0
        else:
            if random.random() < p_m and WDR_list[i] < average_WDR:  #redoslijed oko and?
                solution[i] = 1

    return solution

def crossover(parent_1, parent_2): # TODO: jos dodati random
    child = parent_2[:]
    p_1 = calculate_fitness(parent_1, W)
    p_2 = calculate_fitness(parent_2, W)
    p1 = p_2 / (p_1 + p_2)

    for i in range(len(parent_1)):
        if random.random() < p1:
            child[i] = parent_1[i]

    return child

def genetic_algorithm(E, W, n = 20, p_c = 0.9, max_gen = 100): #TODO: p_c je kompliciraniji
    population = generate_initial_population(n, E, W)

    current_best = find_best_solution(population, W)
    WDR_list = [W[i] / degree(i, E) for i in range(len(E))]
    avg_WDR = sum(WDR_list) / len(E)

    gen = 0
    while gen < max_gen:
        print("generation: ", gen)
        if random.random() < p_c:
            p1 = binary_tournament_selection(population, W)
            p2 = binary_tournament_selection(population, W)

            new_solution = mutate(crossover(p1, p2), WDR_list, avg_WDR)
        else:
            new_solution = generate_random_solution(len(E))

        new_solution = reduction(repair(new_solution, E, W), E, W)

        if new_solution not in population:
            gen += 1

            replace_worst_solution(population, new_solution, W)

            if calculate_fitness(new_solution, W) < calculate_fitness(current_best, W):
                current_best = new_solution

    return current_best


if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename, 'r') as input_file:
        W, E = readData(input_file)

    solution = genetic_algorithm(E, W)
    print(calculate_fitness(solution, W))
    print(check_vertex_cover(solution, E))
