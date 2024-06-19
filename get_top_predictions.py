file_paths = [
    "data/conll03/test/test.txt",
    "data/conll03/train/train.txt",
    "data/conll03/valid/valid.txt",
]
tags = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]
subsets = ["test", "train", "valid"]

for subset in subsets:
    results = {tag: {"Valid": []} for tag in tags}
    for tag in tags:
        prediction_file = f"xlm-roberta/results/{subset}/prediction_counts_top_predictions_{subset}_{tag}_masked.txt"
        max_n = 20
        counter = 1
        original_words = []
        lowercase_words = set()

        for file in file_paths:
            with open(file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 4 and parts[3] == tag:
                        original_words.append(parts[0])
                        lowercase_words.add(parts[0].lower())

        with open(prediction_file, "r") as file:
            lines = file.readlines()[1:]

        for line in lines:
            word = line.split()[0]
            if word.lower() in lowercase_words:
                if counter > max_n:
                    break
                results[tag]["Valid"].append(word)
                counter += 1

    output_file = f"xlm-roberta/results/cleanup_{subset}.txt"
    with open(output_file, "w") as f:
        header_tags = "\t".join([tag for tag in tags])
        f.write(header_tags + "\n")

        max_lines = max(len(results[tag]["Valid"]) for tag in tags)

        for i in range(max_lines):
            line = []
            for tag in tags:
                valid_word = results[tag]["Valid"][i] if i < len(results[tag]["Valid"]) else ""
                line.append(valid_word)
            f.write("\t".join(line) + "\n")
