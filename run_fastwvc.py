import subprocess

args = ["./fastwvc-master/mwvc", "filename", "0", "1", "0"]

args[1] = "./datasets/bio-celegans.dimacs"
popen = subprocess.Popen(args, stdout=subprocess.PIPE)
popen.wait()
output = popen.stdout.read()
print(output)
