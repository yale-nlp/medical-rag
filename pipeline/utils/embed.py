import json
import glob
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def load_config_for_script(config_path: str = "configs.json") -> dict:
    """A simple config loader for standalone script usage."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"[WARN] Config file not found at {config_file}. Using defaults.")
        return {}
    with open(config_file, encoding="utf-8") as f:
        return json.load(f)

def load_all_pages(split_path_pattern: str) -> list:
    """Loads all pages from multiple JSON split files based on a glob pattern."""
    all_pages = []
    json_files = sorted(glob.glob(split_path_pattern))
    if not json_files:
        print(f"[ERROR] No files found for pattern: {split_path_pattern}. Check path in configs.json.")
        return []
    
    print(f"Found {len(json_files)} page files to process.")
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_pages.extend(data.get("pages", []))
    return all_pages

def generate_embeddings_hf(pages: list, model_name: str, device: str, batch_size: int = 32) -> list:
    """
    Generates embeddings for all pages using a HuggingFace model.
    This function loads the model directly, as embedding models are not
    typically served via OpenAI-compatible generation endpoints.
    """
    print(f"[INFO] Loading HuggingFace embedding model '{model_name}' onto device '{device}'.")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load embedding model '{model_name}': {e}")
        return []

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedded_pages = []
    for i in tqdm(range(0, len(pages), batch_size), desc=f"Embedding pages with {model_name}"):
        batch_pages = pages[i:i+batch_size]
        texts = [page.get("markdown", "") for page in batch_pages]
        
        try:
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**encoded)
                token_embeddings = outputs.last_hidden_state
                attention_mask = encoded["attention_mask"]
                
                # Mean pooling logic
                expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
                sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
                embedding_vectors = sum_embeddings / sum_mask
                embedding_vectors = F.normalize(embedding_vectors, p=2, dim=1)

            for j, page in enumerate(batch_pages):
                embedded_pages.append({
                    "index": page.get("index"),
                    "text": texts[j],
                    "embedding": embedding_vectors[j].cpu().tolist()
                })
        except Exception as e:
            print(f"[ERROR] Failed to embed batch starting at index {i}: {e}")

    return embedded_pages

def main():
    """Main function to drive the document embedding process."""
    print("--- Document Embedding Script ---")
    config = load_config_for_script()
    
    # Get settings from config or use defaults
    model_name = config.get("hf_embedding_model", "Alibaba-NLP/gte-Qwen2-7B-instruct")
    device = config.get("embedding_model_device", "cuda" if torch.cuda.is_available() else "cpu")
    
    # The path to data should be relative to the project root.
    # Assuming this script is run from the project root.
    keyword_file_path = Path(config.get("keyword_file", "data/default_keywords.json"))
    project_root = keyword_file_path.parent.parent # Heuristic to find project root
    
    splits_path_pattern = str(project_root / "data/ocr_batches/split_*.json")
    output_path_template = str(project_root / "data/embedded_pages_{model_type}.json")

    pages = load_all_pages(splits_path_pattern)
    if not pages:
        return
    print(f"Loaded {len(pages)} pages from textbook splits.")

    embedded_pages = generate_embeddings_hf(pages, model_name, device)
    
    if embedded_pages:
        model_type_str = "hf_" + model_name.replace("/", "_")
        output_path = output_path_template.format(model_type=model_type_str)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embedded_pages, f, indent=2)
        print(f"Successfully saved {len(embedded_pages)} embedded pages to {output_path}")

if __name__ == "__main__":
    main()