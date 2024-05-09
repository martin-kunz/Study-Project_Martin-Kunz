import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
from data_preprocessing import tokenize_and_align_labels


def main():

    torch.cuda.set_device(2)

    dataset = load_dataset("conll2003")

    model_name = "xlm-roberta-base"
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    model = XLMRobertaForTokenClassification.from_pretrained(model_name, num_labels=9)

    dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True, remove_columns=dataset["train"].column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        do_train=True,
        do_eval=True
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    trainer.train()


if __name__ == "__main__":
    main()
