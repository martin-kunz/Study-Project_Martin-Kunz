import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification, TrainingArguments, pipeline
from data_preprocessing import preprocess_data
import logging
import torch
import numpy as np
from tqdm import tqdm

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check if CUDA is available and set the device
if torch.cuda.is_available():
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"Using device: {device}")
    torch.cuda.set_device(0)
else:
    device = "cpu"
    print("CUDA not available, using CPU")

# Logging setup
logging.basicConfig(
    filename="./logs/evaluation_new-weights.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def compute_metrics(p):
    """
    Compute the evaluation metrics for predictions and labels.

    :param p: A tuple containing predictions and labels.
    :return: A dictionary with precision, recall, f1 score, and accuracy.
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Load CoNLL03 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

# Load XLM-RoBERTa and tokenize data
tokenizer = AutoTokenizer.from_pretrained("./new_model")
tokenized_datasets = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

# Load trained model
model = AutoModelForTokenClassification.from_pretrained("./new_model")
model.to(device)

# Define DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Load metrics
metric = load_metric("seqeval")

# Get label list
label_list = dataset["train"].features["ner_tags"].feature.names

# Training Arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    log_level="info",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Evaluate model
evaluation_results = trainer.evaluate()

# Log/print evaluation checkpoints
logging.info(evaluation_results)
print(evaluation_results)


# Load the token classification pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if device == "cuda" else -1)


# Extract sentences and their corresponding predictions
def get_sentences_with_predictions(dataset, nlp_pipeline, label_list):
    results = []
    for sentence in tqdm(dataset["test"], desc="Processing sentences"):
        tokens = sentence["tokens"]
        ner_tags = [label_list[tag] for tag in sentence["ner_tags"]]
        ner_results = nlp_pipeline(tokens)
        predictions = []
        for token_result in ner_results:
            if isinstance(token_result, list):
                entity = ""
                for sub_token_result in token_result:
                    if isinstance(sub_token_result, dict):
                        entity = sub_token_result["entity"]
                predictions.append(entity)
            elif isinstance(token_result, dict):
                predictions.append(token_result["entity"])
        # Map numeric labels to human-readable labels
        readable_predictions = [label_list[int(label.split("_")[-1])] for label in predictions]
        results.append((tokens, readable_predictions, ner_tags))
    return results


# Get sentences with predictions
sentences_with_predictions = get_sentences_with_predictions(dataset, nlp, label_list)

# Save the results to a file
output_file = "ner_predictions.txt"
with open(output_file, "w") as f:
    for tokens, predictions, ner_tags in sentences_with_predictions:
        f.write(f"Tokens: {' '.join(tokens)}\n")
        f.write(f"Predictions: {' '.join(predictions)}\n")
        f.write(f"True Labels: {' '.join(ner_tags)}\n")
        f.write("\n")
