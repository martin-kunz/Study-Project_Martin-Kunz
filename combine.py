import os
import pandas as pd


def read_file_safely(file_path):
    """

    :param file_path:
    :return:
    """
    data = []
    with open(file_path, "r") as file:
        next(file)
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                data.append(parts)
    return pd.DataFrame(data, columns=["Prediction", "Occ.", "%"])


def combine_txt_files(directory):
    """

    :param directory:
    :return:
    """
    tags = [
        "B-PER",
        "I-PER",
        "B-LOC",
        "I-LOC",
        "B-ORG",
        "I-ORG",
        "B-MISC",
        "I-MISC",
        "O",
    ]
    combined_data = {}

    for tag in tags:
        file_path = os.path.join(
            directory,
            f"prediction_counts_top_predictions_{directory.split('/')[-1]}_{tag}_masked.txt",
        )
        if os.path.exists(file_path):
            combined_data[tag] = read_file_safely(file_path)
        else:
            combined_data[tag] = pd.DataFrame(columns=["Prediction", "Occ.", "%"])

    max_length = max([len(combined_data[tag]) for tag in tags])

    data_rows = []
    for i in range(max_length):
        row = []
        for tag in tags:
            if i < len(combined_data[tag]):
                row.extend(combined_data[tag].iloc[i].values)
            else:
                row.extend(["", "", ""])
        data_rows.append(row)

    # Create the final dataframe
    header_row_1 = [tag for tag in tags for _ in range(3)]
    header_row_2 = ["Prediction", "Occ.", "%"] * len(tags)

    final_df = pd.DataFrame(data_rows, columns=header_row_1)

    output_file = os.path.join(directory, f"combined_predictions_{directory.split('/')[-1]}.tsv")
    with open(output_file, "w") as f:
        f.write("\t".join(header_row_1) + "\n")
        f.write("\t".join(header_row_2) + "\n")
        final_df.to_csv(f, index=False, sep="\t", header=False)

    print(f"Combined data saved to {output_file}")


combine_txt_files("results/train")
combine_txt_files("results/test")
combine_txt_files("results/valid")
