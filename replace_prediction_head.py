import torch
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
import os

# Labels
num_labels = 9

# Load model and tokenizer
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

# Tags with their most common prediction
tag_prediction = {
    "I-PER": "Smith",
    "B-PER": "John",
    "I-LOC": "York",
    "B-LOC": "London",
    "I-ORG": "The",
    "B-ORG": "FC",
    "I-MISC": "Cup",
    "B-MISC": "World",
    "O": "by"
}
token_ids = {label: tokenizer.encode(name, add_special_tokens=False) for label, name in tag_prediction.items()}

# Save old weights/biases
old_weights = model.classifier.weight.detach().clone()
old_bias = model.classifier.bias.detach().clone()

# Initialize new weights/biases
new_weights = old_weights.clone()
new_bias = old_bias.clone()

# Replace weights/bias of the tags with those of the words (e.g.: Replace weights/bias of 'B-PER' with 'John')
for label, ids in token_ids.items():
    idx = list(tag_prediction.keys()).index(label)
    for token_id in ids:
        new_weights[idx] = model.roberta.embeddings.word_embeddings.weight[token_id].detach()
        new_bias[idx] = 0

# Insert weights/biases into model
model.classifier.weight = torch.nn.Parameter(new_weights)
model.classifier.bias = torch.nn.Parameter(new_bias)

# Save new model
output_dir = "./new_model"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved in '{output_dir}'")
