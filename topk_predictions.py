from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
import pandas as pd
from data_preprocessing import get_top_predictions


def main(file):

    # Tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")

    # Read data from file
    with open(f"data/{file}.txt", "r", encoding="utf-8") as f:
        data = f.readlines()

    # Process each sentence and get predictions/probabilities
    results = []
    for sentence in data:
        predictions, probabilities = get_top_predictions(sentence, tokenizer, model)
        results.append({"sentence": sentence.strip(), "top_5_predictions": predictions, "probabilities": probabilities})

    # Convert results to pandas.DataFrame
    result_df = pd.DataFrame(results)

    # Display first 10 results
    print(result_df.head(10))

    # Save results as .csv file
    output_file_path = f"data/top_predictions_{file}.csv"
    result_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main("masked_conll03_test")
    main("masked_conll03_train")
    main("masked_conll03_valid")
