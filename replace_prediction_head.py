import torch
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
import os

# Labels
num_labels = 9

# Load model and tokenizer
model = XLMRobertaForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
print(model)

# Define labels and their most common prediction
labels = ["I-PER", "B-PER", "I-LOC", "B-LOC", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]
tag_prediction = {"I-PER": "Smith", "B-PER": "John", "I-LOC": "York", "B-LOC": "London", "I-ORG": "The", "B-ORG": "FC", "I-MISC": "Cup", "B-MISC": "World", "O": "by"}

# Convert words to token IDs
token_ids = {}
for label in labels:
    name = tag_prediction[label]
    encoded_ids = tokenizer.encode(name, add_special_tokens=False)
    token_ids[label] = encoded_ids

print(f"Token-IDs: {token_ids}")

# Save old weights/biases
old_weights = model.classifier.weight.detach().clone()
old_bias = model.classifier.bias.detach().clone()

# Output the old weights and biases for each output feature
print("Old weights and biases for each output feature:")
for idx, label in enumerate(labels):
    weight = old_weights[idx].detach().numpy()
    bias = old_bias[idx].item()
    print(f"{label}:")
    print(f"  Weight: {weight[0:10]}")
    print(f"  Bias: {bias}")

# Initialize new weights/biases
new_weights = old_weights.clone()
new_bias = old_bias.clone()
print(f"New weights: {new_weights.size()}")

# Replace weights/bias of the tags with those of the words (e.g.: Replace weights/bias of 'B-PER' with 'John')
for label in labels:
    ids = token_ids[label]
    idx = labels.index(label)
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

print(f"-" * 120)

# Output the new weights and biases for each output feature
print("New weights and biases for each output feature:")
for idx, label in enumerate(labels):
    weight = model.classifier.weight[idx].detach().numpy()
    bias = model.classifier.bias[idx].item()
    print(f"{label}:")
    print(f"  Weight: {weight[0:10]}")
    print(f"  Bias: {bias}")
