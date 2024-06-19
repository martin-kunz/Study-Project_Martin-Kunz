import pandas as pd
import ast
from collections import defaultdict
import os


def count_first_predictions(file_path):
    """

    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)

    prediction_counts = defaultdict(int)

    for index, row in df.iterrows():
        first_prediction = ast.literal_eval(row["top_5_predictions"])[0][0]

        prediction_counts[first_prediction] += 1

    sorted_predictions = dict(sorted(prediction_counts.items(), key=lambda item: item[1], reverse=True))

    return sorted_predictions


def save_results_to_txt(prediction_counts, total, output_file):
    """

    :param prediction_counts:
    :param total:
    :param output_file:
    :return:
    """
    with open(output_file, "w") as f:
        # Write header
        f.write("Prediction\tOcc.\t%\n")
        # Write each prediction with its count and percentage
        for prediction, count in prediction_counts.items():
            percentage = (count / total) * 100
            f.write(f"{prediction}\t{count}\t{percentage:.4f} %\n")


def process_all_files(base_dir):
    """

    :param base_dir:
    :return:
    """
    for subset in ["train", "valid", "test"]:
        subset_dir = os.path.join(base_dir, subset)
        output_dir = os.path.join("results", subset)
        os.makedirs(output_dir, exist_ok=True)

        for file_name in os.listdir(subset_dir):
            if file_name.endswith(".csv"):
                file_path = os.path.join(subset_dir, file_name)
                prediction_counts = count_first_predictions(file_path)

                # Total number of predictions
                total_predictions = sum(prediction_counts.values())

                # Output file path
                output_file_path = os.path.join(output_dir, f"prediction_counts_{file_name.replace('.csv', '.txt')}")

                # Save results to the text file
                save_results_to_txt(prediction_counts, total_predictions, output_file_path)

                print(f"Results saved to {output_file_path}")


if __name__ == "__main__":
    base_dir = "data/conll03"
    process_all_files(base_dir)
