"""
Script for extracting and analyzing token embeddings and classifier weights using DistilBERT and XLM-Roberta models.
"""

from transformers import (
    DistilBertModel,
    DistilBertTokenizer,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
)

bert = "distilbert-base-uncased"
xlm_roberta = "xlm-roberta-base"


# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(bert)

# Retrieve the vocabulary
vocab = tokenizer.get_vocab()

# Seach for specific token
token = None

# Sort the vocabulary by indices
sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

# Save the vocabulary to a file
with open("vocab.txt", "w") as f:
    for t, index in sorted_vocab:
        if t == token:
            print(f"{index}: {t}")
        f.write(f"{index}: {t}\n")

# Initialize the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(bert)
model = DistilBertModel.from_pretrained(bert)

# Retrieve the token ID for the desired word
token_id = tokenizer.convert_tokens_to_ids(token)

# Retrieve the embeddings of the model
embeddings = model.get_input_embeddings()

# Extract the weights of the token
token_weights = embeddings.weight[token_id].detach().numpy()

# Output the length of the token weights
print(f"Length for Token '{token}':", len(token_weights))

# Initialize the model with the correct number of classes
num_labels = 9
model = XLMRobertaForTokenClassification.from_pretrained(xlm_roberta, num_labels=num_labels)
tokenizer = XLMRobertaTokenizer.from_pretrained(xlm_roberta)

# Check the classifier weights
print("Classifier weights shape:", model.classifier.weight.shape)
print("Classifier bias shape:", model.classifier.bias.shape)

# Class names
class_labels = ["I-PER", "B-PER", "I-LOC", "B-LOC", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]

# Find index for B-LOC Tag
idx = class_labels.index("B-LOC")

# Extract the weights for the 'B-LOC' class
w = model.classifier.weight[idx]
b = model.classifier.bias[idx]

# Convert the weights and bias to numpy arrays for better readability
w_np = w.detach().numpy()
b_np = b.detach().numpy()

print("Weights for the 'B-LOC' class (Shape):", w_np)
print("Weights for the 'B-LOC' class:", w_np)
print("Bias for the 'B-LOC' class:", b_np)
