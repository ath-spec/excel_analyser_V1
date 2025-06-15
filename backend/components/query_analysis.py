import re
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel, pipeline

from components.embeddings import get_bert_embeddings


def extract_dynamic_tags(query):
    stop_words = set(stopwords.words('english'))
    words = set(re.findall(r'\b\w+\b', query.lower()))
    meaningful_words = words - stop_words
    return meaningful_words

def match_query_to_tags(query, tag_embeddings, threshold=0.5):
    query_embedding = get_bert_embeddings([query])[0]
    matched_columns = []
    matched_values = {}
    additional_tags = extract_dynamic_tags(query)
    all_tags = set()
    
    for col, tag_data in tag_embeddings.items():
        column_tags = set(tag_data["tags"])
        if column_tags & additional_tags:
            all_tags.add(col)

    for col, tag_data in tag_embeddings.items():
        similarities = torch.cosine_similarity(query_embedding, tag_data["embeddings"], dim=1)
        best_score = torch.max(similarities).item()
        if best_score > threshold and col in all_tags:
            matched_columns.append(col)
            for idx, value in enumerate(tag_data["candidates"]):
                value_similarity = similarities[idx].item()
                if value_similarity > threshold:
                    if col not in matched_values:
                        matched_values[col] = []
                    matched_values[col].append((value, value_similarity))
    
    return matched_columns, matched_values

def extract_entities(query):
    entities = {}
    ner_pipeline = pipeline("ner", model="dslim/bert-large-NER", tokenizer="dslim/bert-large-NER")
    ner_results = ner_pipeline(query)
    
    for result in ner_results:
        entity_type = result['entity']
        entity_value = result['word']
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_value)
    
    return entities

def identify_intent(query):
    intents = ["retrieve", "comparison", "analysis", "filter", "summary"]
    zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = zero_shot_classifier(query, candidate_labels=intents)
    return result['labels'][0]

def extract_conditions_from_query(query):
    conditions = {}
    between_matches = re.findall(r"(\w+)\s+between\s+(\d+)\s+and\s+(\d+)", query, re.IGNORECASE)
    for match in between_matches:
        column, lower, upper = match
        conditions[column.lower()] = {"operator": "between", "value": [int(lower), int(upper)]}
    
    other_matches = re.findall(r"(\w+)\s+(above|below|greater than|less than|>=|<=|>|<|=)\s+(\d+)", query, re.IGNORECASE)
    for match in other_matches:
        column, operator, value = match
        operator = operator.lower().replace("above", ">").replace("below", "<").replace("greater than", ">").replace("less than", "<")
        conditions[column.lower()] = {"operator": operator, "value": int(value)}
    
    return conditions

def apply_filters(data, matched_columns, conditions, matched_values, query):
    if matched_columns:
        filtered_data = apply_conditions(data[matched_columns], conditions)
    else:
        filtered_data = data

    return filter_rows_based_on_common_keywords(filtered_data, matched_values, query)

def apply_conditions(data, conditions):
    filtered_data = data.copy()
    for column, condition in conditions.items():
        if column in data.columns:
            operator = condition["operator"]
            value = condition["value"]
            if operator == ">":
                filtered_data = filtered_data[filtered_data[column] > value]
            elif operator == "<":
                filtered_data = filtered_data[filtered_data[column] < value]
            elif operator == ">=":
                filtered_data = filtered_data[filtered_data[column] >= value]
            elif operator == "<=":
                filtered_data = filtered_data[filtered_data[column] <= value]
            elif operator == "=":
                filtered_data = filtered_data[filtered_data[column] == value]
            elif operator == "between":
                lower, upper = value
                filtered_data = filtered_data[(filtered_data[column] >= lower) & (filtered_data[column] <= upper)]
    return filtered_data

def filter_rows_based_on_common_keywords(data, matched_values, query):
    filtered_data = data.copy()
    query_keywords = extract_dynamic_tags(query)
    combined_keywords = set(query_keywords)
    
    for col, matches in matched_values.items():
        combined_keywords.update([word for word, score in matches])
        
    common_keywords = query_keywords.intersection(combined_keywords)
    if not common_keywords:
        print("No common keywords found between the query and dataset values.")
        return data

    print(f"Common Keywords Found: {common_keywords}")
    for col in matched_values.keys():
        if col in filtered_data.columns:
            pattern = '|'.join(re.escape(word) for word in common_keywords)
            filtered_data = filtered_data[filtered_data[col].astype(str).str.contains(pattern, case=False, na=False)]
    return filtered_data
