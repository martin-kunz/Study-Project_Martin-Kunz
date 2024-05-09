from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizer
import torch
from torch.nn.functional import softmax


model = XLMRobertaForTokenClassification.from_pretrained("")
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

sentences = ["", ""]

for sentence in sentences:
   
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = softmax(logits, dim=-1)

    top5_preds = torch.topk(probabilities, 5, dim=-1)

    print(f"Sentence: {sentence}")
    for token_id, top_preds in zip(inputs["input_ids"][0], top5_preds.indices[0]):
        token = tokenizer.decode([token_id])
        tags = [model.config.id2label[idx.item()] for idx in top_preds]
        print(f"\tToken: {token}, Top 5 Predictions: {tags}")
    print("\n")
