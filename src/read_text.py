import os
def getFns(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def writeArrays(arrays, file_name):
    file = open(file_name, "w")
    print("starting array")
    for i in range(len(arrays)):
        for n in range(len(arrays[i])):
            file.write(str(arrays[i][n]) + " ")
        file.write("\n")
    file.close()

def importArray(file_name):
    with open(file_name, "r") as infile:
        array = []
        counter = 0
        try:
            for line in infile:
                line = line.strip()
                array.append(line)
                counter += 1
        except UnicodeDecodeError:
            return("Wrong encoding, line", print(counter))
    return array

def importArrays(file_name):
    with open(file_name, "r") as infile:
        array = [list(line.strip().split()) for line in infile]
    return array


def writeArray(array, name):
    file = open(name, "w")
    for i in range(len(array)):
        file.write(str(array[i]) + "\n")
    file.close()