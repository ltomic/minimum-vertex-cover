import os
import matplotlib.pyplot as plt

folder = "results/"
filelist = sorted([fname for fname in os.listdir(folder)], key = lambda name: name.lower())

fig, ax = plt.subplots()

fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

cell_text = []
for filename in filelist:
    with open(folder + filename, "r") as results_file:
        cell_text.append([results_file.readlines()[1].strip()])

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

for row in cell_text:
    row.append('{0:.2f}'.format(int(row[0]) / int(row[1]) - 1))

table = plt.table(
        cellText = cell_text,
        rowLabels = filelist,
        colLabels = ['genetic', 'fastwvc', 'percentage'],
        loc = 'center',
        colLoc = 'center',
        rowLoc = 'center',
        cellLoc = 'center'
        )

fig.tight_layout()

plt.show()

