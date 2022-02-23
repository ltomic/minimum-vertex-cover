import sys
import random

def readData(data_file, random_weights):
    for line in data_file:
        line = line.strip()

        if line == "":
            continue

        line = line.split(' ')

        if line[0] == "c" or line[0] == "#":
            continue

        if line[0] == "p":
            # line[1] is somekind of FORMAT variable
            n_nodes, n_edges = int(line[2]), int(line[3])

            # vertex weights not listed are assumed to be 1
            # TODO: maybe this is a bad practice
            W = [(i + 2) % 200 if random_weights else 1 for i in range(n_nodes)]
            edges = []

        if line[0] == "v":
            node_index, weight = int(line[1])-1, int(line[2])

            W[node_index] = weight

        if line[0] == "e":
            edges.append((int(line[1])-1, int(line[2])-1))

    if n_edges != len(edges):
        print("Number of edges not matching number of specified edges")
        sys.exit(0)

    return W, edges

def getNumberOfNodesAndEdges(snap_data_file):
    n_nodes = 0
    n_edges = 0

    for line in snap_data_file:
        line = line.strip()

        if line == "":
            continue

        line = line.split(' ')

        if line[0] == '#':
            continue

        n_edges += 1
        n_nodes = max(n_nodes, max(int(line[0])+1, int(line[1])+1))

    snap_data_file.seek(0)
    return n_nodes, n_edges

def snapFormatToDimacsFormat(snap_data_file, dimacs_data_file):
    print("Snap format to dimacs format")
    n_nodes, n_edges = getNumberOfNodesAndEdges(snap_data_file)

    dimacs_data_file.write("p edge " + str(n_nodes) + " " + str(n_edges) + "\n")

    for line in snap_data_file:
        line = line.strip()

        if line == "":
            continue

        line = line.split(' ')

        if line[0] == "#":
            continue

        vertex_a, vertex_b = int(line[0]), int(line[1]) # verifies it is a number
        dimacs_data_file.write("e " + str(vertex_a+1) + " " + str(vertex_b+1) + "\n")

def mtxFormatToDimacsFormat(mtx_data_file, dimacs_data_file, generate_weights = False):
    print("Matrix format to dimacs format")

    for line in mtx_data_file:
        if line.strip()[0] == '%':
            continue

        n_nodes, _, n_edges = line.strip().split(' ')

        break

    print(n_nodes, n_edges)

    dimacs_data_file.write("p edge " + str(n_nodes) + " " + str(n_edges) + "\n")

    if generate_weights == True:
        for i in range(int(n_nodes)):
            dimacs_data_file.write("v " + str(i+1) + " " + str((i+2) % 200) + "\n")

    for line in mtx_data_file:
        line = line.strip().split(' ')

        vertex_a, vertex_b = int(line[0]), int(line[1])
        dimacs_data_file.write("e " + str(vertex_a) + " " + str(vertex_b) + "\n")



if __name__ == "__main__":
    if sys.argv[1] == "snaptodimacs":
        snap_data_filename = sys.argv[2]
        dimacs_data_filename = sys.argv[3]

        with open(snap_data_filename, 'r') as snap_data_file, \
                open(dimacs_data_filename, 'w') as dimacs_data_file:
            snapFormatToDimacsFormat(snap_data_file, dimacs_data_file)

    if sys.argv[1] == "mtxtodimacs":
        mtx_data_filename = sys.argv[2]
        dimacs_data_filename = sys.argv[3]

        with open(mtx_data_filename, 'r') as mtx_data_file, \
                open(dimacs_data_filename, 'w') as dimacs_data_file:
            mtxFormatToDimacsFormat(mtx_data_file, dimacs_data_file)

    if sys.argv[1] == "mtxtodimacs_weights":
        mtx_data_filename = sys.argv[2]
        dimacs_data_filename = sys.argv[3]

        with open(mtx_data_filename, 'r') as mtx_data_file, \
                open(dimacs_data_filename, 'w') as dimacs_data_file:
            mtxFormatToDimacsFormat(mtx_data_file, dimacs_data_file, True)

    if sys.argv[1] == 'readdimacs':
        filename = sys.argv[1]

        with open(filename, 'r') as data_file:
            print(readData(data_file))


