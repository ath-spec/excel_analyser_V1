import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT components
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embeddings(texts):
    inputs = tokenizer_bert(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings

def generate_tags_with_context(data):
    tags = {}
    for col in data.columns:
        column_values = data[col].dropna().astype(str).unique()
        generalized_tags = {col.lower(), col.replace("_", " ").lower()}
        for value in column_values:
            value_tags = {value.lower(), value.replace("_", " ").lower()}
            generalized_tags.update(value_tags)
        generalized_tags = list(generalized_tags)
        embeddings = get_bert_embeddings(generalized_tags)
        tags[col] = {
            "tags": generalized_tags,
            "embeddings": embeddings,
            "candidates": column_values
        }
    return tags
