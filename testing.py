file_path = "data/conll03/test/test.txt"

# Reading the file and extracting the words with I-LOC tag
with open(file_path, "r") as file:
    lines = file.readlines()

i_loc_words = [line.split()[0] for line in lines if len(line.split()) == 4 and line.split()[3] == "B-PER"]

print(len(i_loc_words))
print(len(set(i_loc_words)))
