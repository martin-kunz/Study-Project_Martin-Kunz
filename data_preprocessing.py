import torch
import torch.nn.functional as F


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


def replace_tags_with_mask(input_file, output_file, tags_to_mask, mask_token="<mask>"):
    """
    Replaces words with specified NER tags with a mask token and saves the sentences in a single line without additional information.

    :param input_file: Path to the input file in CoNLL03 format.
    :param output_file: Path to the output file where masked sentences will be saved.
    :param tags_to_mask: List of NER tags to be replaced with the mask token.
    :param mask_token: The token to replace specified Named Entity tags with. Default is "<mask>".
    :return: None
    """
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    sentences = []
    current_sentence = []

    for line in lines:
        if line.strip() and not line.startswith("-DOCSTART-"):
            parts = line.split()
            if len(parts) == 4:
                word, pos, chunk, ne_tag = parts
                if ne_tag in tags_to_mask:
                    word = mask_token
                current_sentence.append(word)
            else:
                if current_sentence:
                    sentences.append(" ".join(current_sentence))
                    current_sentence = []
        else:
            if current_sentence:
                sentences.append(" ".join(current_sentence))
                current_sentence = []

    if current_sentence:
        sentences.append(" ".join(current_sentence))

    with open(output_file, "w", encoding="utf-8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")


def get_top_predictions(sentence, tokenizer, model):
    """
    Get the top 5 predictions and their probabilities for masked tokens in a sentence.

    :param sentence: The input sentence containing masked tokens.
    :param tokenizer: The tokenizer used to process the text.
    :param model: The pre-trained model used for making predictions.
    :return: A tuple containing two lists: top 5 predictions and their probabilities for each masked token.
    """
    inputs = tokenizer(sentence, return_tensors="pt")
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

        # Convert token ids to words and store results
        top_5_words = [tokenizer.decode([token]).strip() for token in top_5_tokens]
        top_5_predictions.append(top_5_words)
        top_5_probabilities.append(top_5_probs)

    return top_5_predictions, top_5_probabilities