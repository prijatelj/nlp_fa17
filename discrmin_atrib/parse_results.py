from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir("results/") if isfile(join("results/", f))]

with open("results/total_results.csv", 'w') as csv:
    csv.write("file,accuracy\n")
    for result in onlyfiles:
        with open("results/" + result, 'r') as f:
            content = f.readlines()
            for line in content:
                if "acc" in line:
                    line = line.split(" = ")
                    csv.write(result + "," + line[1] + "\n")

