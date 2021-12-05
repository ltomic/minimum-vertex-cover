# run with
# python minimumVertexCover.py ./data/facebook_combined.txt
import random

import sys

def degree(x, E):  #prebaciti E u globalnu varijablu? isto za W
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

def find_vertex_with_highest_weight_degree_ratio(covered_vertices, W, E):
    sol = list(covered_vertices)[0] # TODO: ovo treba nekako bolje
    max_ratio = W[sol] / degree(sol, E)

    for i in covered_vertices:
        ratio = W[i] / degree(i, E)

        if ratio > max_ratio:
            max_ratio = ratio
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



def generate_random_solution(num_vertex): #generira pocetno rjesenje
    return [random.randint(0, 1) if degree(i, E) > 0 else 0 for i in range(num_vertex)]

if __name__ == "__main__":
    filename = sys.argv[1]

    with open(filename, 'r') as input_file:
        W, E = readData(input_file)

    state = generate_random_solution(len(W))
    print(state.count(1))
    state = reduction(state, E, W)
    print(state.count(1))
