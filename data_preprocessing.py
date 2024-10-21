import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def preprocess_data(examples, tokenizer):
    """
    Preprocesses the data for token classification tasks by tokenizing the input and aligning the NER tags.

    :param examples: A dictionary containing tokenized text ("tokens") and corresponding NER tags ("ner_tags").
    :param tokenizer: The tokenizer used to tokenize the input text.
    :return: A dictionary with tokenized inputs and padded labels for each token.
    """
    # Tokenize the input tokens and prepare for token classification
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Assign -100 for non-aligned tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Use the actual label for the first subtoken
                label_ids.append(label[word_idx])
            else:
                # Assign -100 for subsequent subtokens (to ignore them in loss computation)
                label_ids.append(-100)
            previous_word_idx = word_idx

        # Pad the labels to the model's maximum sequence length
        label_ids += [-100] * (tokenizer.model_max_length - len(label_ids))
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def replace_tags_with_mask(file_path: str, tag: str, output_path: str, mask_token: str = "<mask>") -> None:
    """
    Reads a file, masks words with a specified NER tag, splits sentences with multiple tag occurrences,
    and writes the result to an output file.

    :param file_path: The path to the input file containing sentences and tags.
    :param tag: The NER tag to be masked in the sentences.
    :param output_path: The path to save the processed output file.
    :param mask_token: The token used to replace the masked words. Default is "<mask>".
    :return: None
    """
    # Read the entire content from the input file
    with open(file_path, "r") as file:
        content = file.read()

    # Split the content by sentence (assuming double newlines between sentences)
    sentences = content.split("\n\n")
    result = []

    for sentence in sentences:
        words = sentence.split("\n")
        masked_sentence = []
        tag_count = 0

        # Count the occurrences of the specified tag in the sentence
        for word in words:
            if word:
                parts = word.split()
                if len(parts) > 3 and parts[3] == tag:
                    tag_count += 1

        # If the tag is found in the sentence, proceed with masking
        if tag_count > 0:
            split_triggered = False
            for word in words:
                if word:
                    parts = word.split()
                    if len(parts) > 3 and parts[3] == tag:
                        if split_triggered:
                            # If another tag is found, split the sentence
                            result.append(" ".join(masked_sentence))
                            masked_sentence = [mask_token]
                        else:
                            masked_sentence.append(mask_token)
                            split_triggered = True
                    else:
                        masked_sentence.append(parts[0])
            if masked_sentence:
                result.append(" ".join(masked_sentence))

    # Write the masked sentences to the output file
    with open(output_path, "w") as output_file:
        for sentence in result:
            output_file.write(sentence + "\n")


def get_top_predictions(sentence, tokenizer, model, device):
    """
    Retrieve the top 5 predictions and their corresponding probabilities for each masked token in a sentence.

    :param sentence: The input sentence containing masked tokens.
    :param tokenizer: The tokenizer used to process the sentence.
    :param model: The pre-trained model used to predict the masked tokens.
    :param device: The device (CPU or CUDA) to run the model on.
    :return: Two lists - top 5 predictions and their corresponding probabilities for each masked token.
    """
    # Tokenize the input sentence and move the tensors to the specified device
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    # Find the indices of the masked tokens
    mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    if len(mask_token_indices) == 0:
        return ["No mask token found"], [0] * 5

    top_5_predictions = []
    top_5_probabilities = []

    for mask_token_index in mask_token_indices:
        # Perform a forward pass to get the model's logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the softmax probabilities for the masked token
        softmax_logits = F.softmax(logits[0, mask_token_index], dim=-1)

        # Retrieve the top 5 predictions and their probabilities
        top_5_tokens = torch.topk(softmax_logits, 5, dim=-1).indices.tolist()
        top_5_probs = torch.topk(softmax_logits, 5, dim=-1).values.tolist()

        # Decode the token IDs to words and store them with probabilities
        top_5_words = [tokenizer.decode([token]).strip() for token in top_5_tokens]
        top_5_predictions.append(top_5_words)
        top_5_probabilities.append(top_5_probs)

    return top_5_predictions, top_5_probabilities


def process_file(file_path, output_dir, device, tokenizer, model):
    """
    Processes a file containing sentences, applies the model to get predictions for masked tokens,
    and saves the results to a CSV file.

    :param file_path: Path to the input file containing sentences.
    :param output_dir: Directory where the output CSV file will be saved.
    :param device: The device (CPU or CUDA) to run the model on.
    :param tokenizer: The tokenizer used to tokenize the input sentences.
    :param model: The pre-trained model used to predict the masked tokens.
    :return: None
    """
    # Read the input file line by line
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    results = []

    # Process each sentence in the file
    for sentence in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        predictions, probabilities = get_top_predictions(sentence, tokenizer, model, device)
        results.append(
            {
                "sentence": sentence.strip(),
                "top_5_predictions": predictions,
                "probabilities": probabilities,
            }
        )

    # Convert the results to a pandas DataFrame
    result_df = pd.DataFrame(results)

    # Save the DataFrame as a CSV file
    output_file_name = f"predictions_{os.path.basename(file_path).replace('.txt', '')}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    result_df.to_csv(output_file_path, index=False)

    print(f"Results saved to {output_file_path}")
