from transformers import XLMRobertaTokenizer, XLMRobertaForMaskedLM
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from data_preprocessing import process_file
import os
import torch


def main():

    base_dir = "data/conll03"
    subsets = ["train", "valid", "test"]

    if torch.cuda.is_available():
        device = "cuda"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"Using device: {device}")
        torch.cuda.set_device(0)
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Tokenizer and model
    # tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    # model = XLMRobertaForMaskedLM.from_pretrained("xlm-roberta-base")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
    print(model)
    model.to(device)

    # Iterate over train, valid, and test directories
    for subset in subsets:
        subset_dir = os.path.join(base_dir, subset)
        output_dir = subset_dir

        # Process each file in the subset directory
        for file_name in os.listdir(subset_dir):
            if file_name.endswith(".txt") and "masked" in file_name:
                file_path = os.path.join(subset_dir, file_name)
                process_file(file_path, output_dir, device, tokenizer, model)


if __name__ == "__main__":
    main()
