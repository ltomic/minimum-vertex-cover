# run with
# python minimumVertexCover.py ./data/facebook_combined.txt
import random

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
    WDR = []
    averageWDR = 0
    v = len(E)                  #MOZDA u readData returnat i num_vertex pa to u mainu cuvat i u ovim situacijama zvat
    
    for i in range(v):
        x = W[i] / degree(i, E)
        WDR.append(x)
        averageWDR += x

    averageWDR/=v
        
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

def find_vertex_with_highest_weight_degree_ratio(covered_vertices, WDR, E):
    sol = list(covered_vertices)[0] # TODO: ovo treba nekako bolje
    max_ratio = W[sol] / degree(sol, E)

    for i in covered_vertices:

        if WDR[i] > max_ratio:
            max_ratio = WDR[i]
            sol = i              #statisticki najgori vrh u rjesenju

    return sol


# izbaciti visak iz rjesenja
def reduction(state, E, W):
    num_vertex = len(state)   #broj vrhova

    # vertex whose edges are covered by neighbor vertex
    covered_vertices = find_covered_vertices(state, E)

    # probability with which the vertex having the
    # maximum ratio of weight to degree is selected
    # TODO: namjestiti kao parametar
    p_sc = 0.5
    while len(covered_vertices) != 0:
        if random.random() > p_sc:
            vertex_to_remove = find_vertex_with_highest_weight_degree_ratio(covered_vertices, W, E)
        else:                       #ovako Singh izbaci random vrh ako ne izbaci najgori
            # TODO: mozda promjeniti covered_vertices u list jer ovo traje O(n)
            vertex_to_remove = random.choice(list(covered_vertices))

        covered_vertices.remove(vertex_to_remove)
        state[vertex_to_remove] = 0

        covered_vertex = find_covered_vertices(state, E)

    return state

def find_vertex_not_in_state_with_largest_uncovered_edge_weight_ratio(state,
        candidate_vertices, E, W):
    sol = -1
    max_ratio = -1

    for i in candidate_vertices:
        ratio = calculate_uncovered_edges(state, E, i) / W[i]

        if ratio > max_ratio: # TODO: jos dodati tu random
            max_ratio = ratio
            sol = i

    return sol

def check_vertex_cover(state, E):
    for i in range(len(state)):
        if state[i]==0:
            for j in E[i]:
                if state[j]==0:
                    return False
                
    return True

def first_repair_heuristic(state, E, W):
    vertices_not_in_solution = set([i for i in range(len(state)) if state[i] == 0])

    while check_vertex_cover(state, E) == False:
        v = find_vertex_with_largest_uncovered_edge_weight_ratio(state,
                vertices_not_in_solution, E, W)

        state[v] = 1
        vertices_not_in_solution.remove(v)

    return state

def second_repair_heuristic(state, E, W):
    vertices_not_in_solution = set([i for i in range(state) if state[i] == 0])

    while check_vertex_cover(state, E) == False:
        v = random.choice(list(vertices_not_in_solution))
        A = [i for i in E[v] if state[i] == 0]
        A.append(v)

        s = find_vertex_with_largest_uncovered_edge_weight_ratio(state,
                A, E, W)

        state[v] = 1
        vertices_not_in_solution.remove(v)

    return state

def repair(state, E, W, p_h = 0.5):
    if random.random() < p_h:
        return first_repair_heuristic(state, E, W)

    return second_repair_heuristic(state, E, W)

def generate_random_solution(num_vertex): #generira pocetno rjesenje
    return [random.randint(0, 1) if degree(i, E) > 0 else 0 for i in range(num_vertex)]

def generate_solution(E, W):
    num_vertex = len(E)

    return reduction(repair(generate_random_solution(num_vertex), E, W), E, W)

def generate_initial_population(n, E, W):
    population = []

    while len(population) < n:
        new_solution = generate_solution(E, W)

        if new_solution not in population: # TODO: ovo bi moglo biti sporo
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

    return best_score

def find_worst_solution(population, W):
    worst_solution = population[0]
    worst_score = calculate_fitness(worst_solution, W)

    for solution in population:
        current_score = calculate_fitness(solution, W)

        if current_score > best_score:
            worst_solution = solution
            worst_score = current_score

    return worst_score

def replace_worst_solution(population, new_solution, W):
    worst_solution_index = find_worst_solution(population)

    population[worst_solution_index] = new_solution
    
def mutate(solution, WDR, averageWDR, p_m = 0.1): #TODO: p_m=???
    for i in len(solution):
        if solution[i]==1:
            if random.random() < p_m:
                solution[i]=0
        else:
            if random.random() < p_m and WDR[i]<averageWDR:  #redoslijed oko and?
                solution[i]=1
                
    return solution

def crossover(parent1, parent2):
    child = p_2[:]
    p_1 = calculate_fitness(parent1, W)
    p_2 = calculate_fitness(parent2, W)
    p1 = p_2 / (p_1 + p_2)
    for i in range(len(p_1)):
        if random.random() < p1:
            child[i] = parent1[i]

    return child

def genetic_algorithm(E, W, n = 10, p_c = 0.5, max_gen = 100): #TODO: p_c je kompliciraniji
    population = generate_initial_population(n, E, W)

    current_best = find_best_solution(population, W)

    gen = 0
    while generation < max_gen:
        if random.random() < p_c:
            p1, p2 = binary_tournament_selection()

            new_solution = mutate(crossover(p1, p2))
        else:
            new_solution = generate_random_solution(len(E))

        new_solution = reduction(repair(new_solution, E, W), E, W)

        if new_solution not in population: # TODO: mozda sporo
            gen += 1

            replace_worst_solution(population, new_solution)

            if calculate_fitness(new_solution, W) < calculate_fitness(current_best):
                current_best = new_solution

    return current_best


if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename, 'r') as input_file:
        W, E = readData(input_file)

    state = generate_random_solution(len(W))
    print(state.count(1))
    state = reduction(state, E, W)
    print(state.count(1))
