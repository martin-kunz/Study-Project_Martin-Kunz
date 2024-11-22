import os
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, DataCollatorForTokenClassification, TrainingArguments, pipeline
from data_preprocessing import preprocess_data
import logging
import torch
import numpy as np

name = "./final_model"
log_path = "./logs/evaluation.log"

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
    filename=log_path,
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
tokenizer = AutoTokenizer.from_pretrained(name)
tokenized_datasets = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)

# Load trained model
model = AutoModelForTokenClassification.from_pretrained(name)
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
