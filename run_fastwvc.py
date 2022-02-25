import subprocess
import os
import datetime as dt

args = ["./fastwvc-master/mwvc", "filename", "0", "1", "0"]

folder = "datasets/"
filelist = sorted([fname for fname in os.listdir(folder)], key = lambda name: name.lower())

date = dt.datetime.now()
date_text = date.strftime("%d-%m-%Y-%H-%M-%S")
with open("fastwvc_results_" + date_text + ".txt", "w") as results:
    for filename in filelist:
        args[1] = folder + filename
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        print(output.decode("utf-8"))
        results.write(output.decode("utf-8"))
