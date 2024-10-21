import os
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm


def preprocess_data(examples, tokenizer):
    """
    Preprocess the data for token classification tasks.
    :param examples: A dictionary containing tokenized text and corresponding NER tags.
    :param tokenizer: The tokenizer used to process the text.
    :return: A dictionary with tokenized inputs and padded labels.
    """
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
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        # Pad the labels to the max length
        label_ids += [-100] * (tokenizer.model_max_length - len(label_ids))
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def replace_tags_with_mask(file_path: str, tag: str, output_path: str, mask_token: str = "<mask>") -> None:
    """
    Reads a file, masks words with a specified tag, splits sentences if multiple tags are found,
    and writes the result to an output file.
    :param file_path: The path to the input file.
    :param tag: The tag to be masked.
    :param output_path: The path to the output file.
    :param mask_token: The token used to replace the masked words. Default is "<mask>".
    :return: None
    """
    with open(file_path, "r") as file:
        content = file.read()

    sentences = content.split("\n\n")
    result = []

    for sentence in sentences:
        words = sentence.split("\n")
        masked_sentence = []
        tag_count = 0
        for word in words:
            if word:
                parts = word.split()
                if len(parts) > 3 and parts[3] == tag:
                    tag_count += 1

        if tag_count > 0:
            split_triggered = False
            for word in words:
                if word:
                    parts = word.split()
                    if len(parts) > 3 and parts[3] == tag:
                        if split_triggered:
                            result.append(" ".join(masked_sentence))
                            masked_sentence = [mask_token]
                        else:
                            masked_sentence.append(mask_token)
                            split_triggered = True
                    else:
                        masked_sentence.append(parts[0])
            if masked_sentence:
                result.append(" ".join(masked_sentence))

    with open(output_path, "w") as output_file:
        for sentence in result:
            output_file.write(sentence + "\n")


def get_top_predictions(sentence, tokenizer, model, device):
    """
    Get the top 5 predictions and their probabilities for masked tokens in a sentence.
    :param sentence: The input sentence containing masked tokens.
    :param tokenizer: The tokenizer used to process the text.
    :param model: The pre-trained model used for making predictions.
    :param device: The device to run the model on (cpu or cuda).
    :return: A tuple containing two lists: top 5 predictions and their probabilities for each masked token.
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    mask_token_indices = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    if len(mask_token_indices) == 0:
        return ["No mask token found"], [0] * 5

    top_5_predictions = []
    top_5_probabilities = []

    for mask_token_index in mask_token_indices:
        # Forward pass to get logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get top 5 predictions
        softmax_logits = F.softmax(logits[0, mask_token_index], dim=-1)
        top_5_tokens = torch.topk(softmax_logits, 5, dim=-1).indices.tolist()
        top_5_probs = torch.topk(softmax_logits, 5, dim=-1).values.tolist()

        # Convert token ids to words and store checkpoints
        top_5_words = [tokenizer.decode([token]).strip() for token in top_5_tokens]
        top_5_predictions.append(top_5_words)
        top_5_probabilities.append(top_5_probs)

    return top_5_predictions, top_5_probabilities


def process_file(file_path, output_dir, device, tokenizer, model):
    """
    :param file_path:
    :param output_dir:
    :param device:
    :param tokenizer:
    :param model:
    :return:
    """
    # Read data from file
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.readlines()

    # Process each sentence and get predictions/probabilities
    results = []
    for sentence in tqdm(data, desc=f"Processing {os.path.basename(file_path)}"):
        predictions, probabilities = get_top_predictions(sentence, tokenizer, model, device)
        results.append(
            {
                "sentence": sentence.strip(),
                "top_5_predictions": predictions,
                "probabilities": probabilities,
            }
        )

    # Convert results to pandas.DataFrame
    result_df = pd.DataFrame(results)

    # Save results as .csv file
    output_file_name = f"predictions_{os.path.basename(file_path).replace('.txt', '')}.csv"
    output_file_path = os.path.join(output_dir, output_file_name)
    result_df.to_csv(output_file_path, index=False)
