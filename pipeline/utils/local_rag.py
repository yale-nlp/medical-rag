import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm import tqdm
import logging

_EMBEDDING_MODEL = None
_EMBEDDING_TOKENIZER = None
_LOADED_MODEL_NAME_DEVICE = None

def _get_embedding_model(model_name: str, device: str):
    """
    Loads and caches the HuggingFace embedding model and tokenizer.
    Imports are deferred to prevent premature CUDA initialization in the main process.
    """
    global _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER, _LOADED_MODEL_NAME_DEVICE
    
    model_device_key = f"{model_name}@{device}"
    if _LOADED_MODEL_NAME_DEVICE == model_device_key:
        return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER

    from transformers import AutoTokenizer, AutoModel
    import torch

    logging.info(f"Loading query embedding model '{model_name}' onto device '{device}'.")
    try:
        _EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _EMBEDDING_MODEL = AutoModel.from_pretrained(model_name).to(device).eval()
        if _EMBEDDING_TOKENIZER.pad_token is None:
            _EMBEDDING_TOKENIZER.pad_token = _EMBEDDING_TOKENIZER.eos_token
        _LOADED_MODEL_NAME_DEVICE = model_device_key
        return _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER
    except Exception as e:
        logging.error(f"Failed to load query embedding model '{model_name}': {e}", exc_info=True)
        _EMBEDDING_MODEL, _EMBEDDING_TOKENIZER, _LOADED_MODEL_NAME_DEVICE = None, None, None
        return None, None

def embed_queries(queries: list, model_name: str, device: str, batch_size: int = 64) -> list:
    """Embeds a list of queries using a specified HuggingFace model."""
    if not queries:
        return []

    model, tokenizer = _get_embedding_model(model_name, device)
    if not model or not tokenizer:
        return [np.array([]) for _ in queries]

    import torch
    import torch.nn.functional as F

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(queries), batch_size), desc=f"Embedding queries with {model_name}", leave=False):
            batch_queries = queries[i:i+batch_size]
            encoded = tokenizer(batch_queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            
            expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
            sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
            embedding_vectors = sum_embeddings / sum_mask
            embedding_vectors = F.normalize(embedding_vectors, p=2, dim=1)
            
            all_embeddings.extend([vec.cpu().numpy() for vec in embedding_vectors])
            
    return all_embeddings

def retrieve_top_index(query_embedding: np.ndarray, page_embeddings: list) -> int:
    """Retrieves the list index of the most similar page embedding."""
    if query_embedding.size == 0 or not page_embeddings:
        return 0
    
    all_page_vectors = np.array([p["embedding"] for p in page_embeddings])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), all_page_vectors).flatten()
    return int(np.argmax(similarities))

def get_surrounding_pages(target_index: int, all_pages: list, window_size: int = 1) -> list:
    """Gets the main page and its surrounding pages."""
    if not all_pages or not (0 <= target_index < len(all_pages)):
        return []

    start_idx = max(0, target_index - window_size)
    end_idx = min(len(all_pages), target_index + window_size + 1)
    
    return [{k: v for k, v in page.items() if k != "embedding"} for page in all_pages[start_idx:end_idx]]

def load_embeddings(path: str) -> list:
    """Loads page embeddings from a JSON file."""
    path_obj = Path(path)
    if not path_obj.exists():
        logging.error(f"Embedding file not found: {path_obj}")
        return []
    try:
        with open(path_obj, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from embedding file: {path_obj}")
        return []