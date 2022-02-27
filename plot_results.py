import os
import matplotlib.pyplot as plt
import csv

fig, ax = plt.subplots()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

cell_text = []
graphs = []
with open("results/results.txt", "r") as results_file:
    results = csv.reader(results_file, delimiter = ',')
    for row in results:
        if row[0] == 'graph':
            continue
        print(row)
        graphs.append(row[0])
        cell_text.append([row[1], '{0:.3f}'.format(float(row[4]))])
        #cell_text.append([results_file.readlines()[1].strip()])

results = []
with open("fastwvc_results.txt", "r") as results_file:
    for line in results_file:
        result = line.split(',')
        result[0] = result[0][result[0].index('/')+1:]
        results.append(result)
    results = sorted(results, key = lambda name: name[0])
    print(results)
    for i, result in enumerate(results):
        cell_text[i].append(result[1])
        cell_text[i].append('{0:.3f}'.format(float(result[2])))

for row in cell_text:
    row.append('{0:.3f}'.format(int(row[0]) / int(row[2]) - 1))

table = plt.table(
        cellText = cell_text,
        rowLabels = graphs,
        colLabels = ['genetic', 'time', 'fastwvc', 'time', 'percentage'],
        loc = 'center',
        colLoc = 'center',
        rowLoc = 'center',
        cellLoc = 'center'
        )

fig.tight_layout()

plt.show()

