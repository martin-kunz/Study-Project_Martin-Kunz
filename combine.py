import os
import pandas as pd


def read_file(file_path):
    """
    Reads a tab-separated text file, skips the header, and converts it into a pandas DataFrame.

    :param file_path: Path to the text file containing predictions, occurrences, and percentages.
    :return: A pandas DataFrame with columns ["Prediction", "Occ.", "%"].
    """
    data = []
    with open(file_path, "r") as file:
        # Skip the header row
        next(file)
        for line in file:
            # Split the line by tabs and append it to the data if it has 3 parts (Prediction, Occurrence, Percentage)
            parts = line.strip().split("\t")
            if len(parts) == 3:
                data.append(parts)
    return pd.DataFrame(data, columns=["Prediction", "Occ.", "%"])


def combine_txt_files(directory):
    """
    Combines prediction data from multiple tag-specific text files into a single DataFrame and saves it as a TSV file.

    :param directory: The directory where the text files for different tags (I-PER, B-LOC, etc.) are located.
    :return: Saves the combined data in a TSV file with a structure of multiple columns for each tag.
    """
    # List of tags to be processed, each representing different entities
    tags = ["I-PER", "B-PER", "I-LOC", "B-LOC", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]

    combined_data = {}

    for tag in tags:
        # Construct the file path for each tag-specific text file
        file_path = os.path.join(
            directory,
            f"count_predictions_{directory.split('/')[-1]}_{tag}_masked.txt",
        )
        if os.path.exists(file_path):
            # If the file exists, read its contents into a DataFrame
            combined_data[tag] = read_file(file_path)
        else:
            # If the file doesn't exist, create an empty DataFrame with the appropriate columns
            combined_data[tag] = pd.DataFrame(columns=["Prediction", "Occ.", "%"])

    # Find the maximum length among all DataFrames to ensure all rows align correctly
    max_length = max([len(combined_data[tag]) for tag in tags])

    data_rows = []
    for i in range(max_length):
        row = []
        for tag in tags:
            if i < len(combined_data[tag]):
                # If the tag DataFrame has data for this index, append its values to the row
                row.extend(combined_data[tag].iloc[i].values)
            else:
                # If the index exceeds the length of the DataFrame, append empty strings
                row.extend(["", "", ""])
        data_rows.append(row)

    # Create the final DataFrame with headers that reflect the tags and their associated columns
    header_row_1 = [tag for tag in tags for _ in range(3)]
    header_row_2 = ["Prediction", "Occ.", "%"] * len(tags)

    final_df = pd.DataFrame(data_rows, columns=header_row_1)

    # Define the output file path and write the combined data to a TSV file
    output_file = os.path.join(directory, f"top_predictions_{directory.split('/')[-1]}.tsv")
    with open(output_file, "w") as f:
        # Write the multi-level headers to the file
        f.write("\t".join(header_row_1) + "\n")
        f.write("\t".join(header_row_2) + "\n")
        # Save the DataFrame without headers (since they were manually written)
        final_df.to_csv(f, index=False, sep="\t", header=False)

    print(f"Combined data saved to {output_file}")


# Combine prediction files for the train, test, and valid datasets
combine_txt_files("results/train")
combine_txt_files("results/test")
combine_txt_files("results/valid")
