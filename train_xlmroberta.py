import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from data_preprocessing import preprocess_data
import logging

model_name = "xlm-roberta-base"

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Create necessary directories
directories = ["./logs", "./checkpoints", "./final_model", "./results"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Logging setup
logging.basicConfig(
    filename="./logs/training.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Load CoNLL03 dataset
dataset = load_dataset("conll2003")

# Load XLM-RoBERTa and tokenize data
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_datasets = dataset.map(lambda x: preprocess_data(x, tokenizer), batched=True)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9)
print(model)

# Define DataCollator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./checkpoints",
    save_steps=1000,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    log_level="info",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train/save model
trainer.train()
trainer.save_model("./final_model")
