import pandas as pd
import ast
from collections import defaultdict
import os


def count_first_predictions(file_path):
    """
    Reads a CSV file containing predictions and counts how many times each first prediction appears.

    :param file_path: Path to the CSV file containing top-5 predictions as one of the columns.
    :return: A dictionary with predictions as keys and their occurrence counts as values, sorted by count in descending order.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Dictionary to store counts of first predictions
    prediction_counts = defaultdict(int)

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the first prediction from the "top_5_predictions" column (assumed to be a list of tuples in string form)
        first_prediction = ast.literal_eval(row["top_5_predictions"])[0][0]

        # Increment the count for this prediction
        prediction_counts[first_prediction] += 1

    # Sort the predictions by their counts in descending order
    sorted_predictions = dict(sorted(prediction_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_predictions


def save_results_to_txt(prediction_counts, total, output_file):
    """
    Saves the counts of each prediction to a text file, along with their respective percentages.

    :param prediction_counts: Dictionary of predictions and their counts.
    :param total: Total number of predictions.
    :param output_file: Path where the output text file should be saved.
    """
    with open(output_file, "w") as f:
        # Write header for the output file
        f.write("Prediction\tOcc.\t%\n")

        # Write each prediction along with its count and the percentage of total occurrences
        for prediction, count in prediction_counts.items():
            percentage = (count / total) * 100
            f.write(f"{prediction}\t{count}\t{percentage:.4f} %\n")


def process_all_files(base_dir):
    """
    Processes all CSV files in the given directory for different subsets (train, valid, test) and
    saves the prediction counts for each file into respective output text files.

    :param base_dir: Base directory containing subdirectories ("train", "valid", "test") with CSV files.
    """
    # Loop through each subset (train, valid, test)
    for subset in ["train", "valid", "test"]:
        # Create the path to the subset directory
        subset_dir = os.path.join(base_dir, subset)

        # Create an output directory for the subset results
        output_dir = os.path.join("results", subset)
        os.makedirs(output_dir, exist_ok=True)

        # Loop through all CSV files in the subset directory
        for file_name in os.listdir(subset_dir):
            if file_name.endswith(".csv"):
                # Construct the full file path
                file_path = os.path.join(subset_dir, file_name)

                # Count the first predictions in the file
                prediction_counts = count_first_predictions(file_path)

                # Calculate the total number of predictions
                total_predictions = sum(prediction_counts.values())

                # Define the output file path
                output_file_path = os.path.join(output_dir, f"count_{file_name.replace('.csv', '.txt')}")

                # Save the prediction counts and percentages to the output file
                save_results_to_txt(prediction_counts, total_predictions, output_file_path)

                print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    # Base directory where the data is stored
    base_dir = "data/conll03"

    # Process all files in the base directory for each subset
    process_all_files(base_dir)
