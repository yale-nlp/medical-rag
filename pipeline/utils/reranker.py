import asyncio
import logging
from typing import Dict, Any, List, Union
from llm_services import LocalLLMService

def _format_document(doc: Union[str, Dict[str, Any]]) -> str:
    """Converts a document (string or dict) into a single text block."""
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        return f"Title: {title}\nAbstract: {abstract}".strip()
    return ""

async def _rerank_with_qwen_local(
    query: str,
    documents: List[Dict[str, Any]],
    llm_service: LocalLLMService,
    instruction: str
) -> List[Dict[str, Any]]:
    """
    Reranks documents by calling the async rerank method of the LocalLLMService.
    This function now creates a NEW list of scored documents instead of modifying the input.
    """
    if not documents:
        return []
        
    doc_texts = [_format_document(doc) for doc in documents]
    scores = await llm_service.rerank(query, doc_texts, instruction)
    
    scored_documents = []
    for i, doc in enumerate(documents):
        new_doc = doc.copy()
        new_doc["rerank_score"] = scores[i] if i < len(scores) else 0.0
        scored_documents.append(new_doc)
        
    return scored_documents

def _rerank_with_cross_encoder(
    query: str,
    documents: List[Dict[str, Any]],
    model_name: str,
    device: str
) -> List[Dict[str, Any]]:
    """Reranks documents using a standard CrossEncoder model."""
    from sentence_transformers import CrossEncoder
    
    try:
        reranker = CrossEncoder(model_name, device=device, max_length=512)
    except Exception as e:
        logging.error(f"Failed to load CrossEncoder model '{model_name}': {e}", exc_info=True)
        return [{**doc, "rerank_score": -1.0} for doc in documents]

    doc_texts = [_format_document(doc) for doc in documents]
    sentence_pairs = [[query, doc_text] for doc_text in doc_texts]
    
    try:
        scores = reranker.predict(sentence_pairs, show_progress_bar=False)
        return [{**documents[i], "rerank_score": float(score)} for i, score in enumerate(scores)]
    except Exception as e:
        logging.error(f"CrossEncoder prediction failed: {e}", exc_info=True)
        return [{**doc, "rerank_score": -1.0} for doc in documents]

async def rerank_candidates_async(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
    model_name: str,
    device: str,
    instruction_task: str,
    llm_service: LocalLLMService = None
) -> List[Dict[str, Any]]:
    """
    Asynchronously dispatches reranking, sorts the results, and returns the top K documents.
    """
    if not documents:
        return []

    is_qwen_reranker = "qwen" in model_name.lower() and "reranker" in model_name.lower()
    
    if is_qwen_reranker:
        if not llm_service:
            logging.error("A LocalLLMService instance is required for Qwen reranking.")
            return []
        logging.debug(f"Reranking with Qwen model '{model_name}' via local vLLM.")
        scored_documents = await _rerank_with_qwen_local(query, documents, llm_service, instruction_task)
    else:
        logging.debug(f"Reranking with standard CrossEncoder model '{model_name}'.")
        loop = asyncio.get_running_loop()
        scored_documents = await loop.run_in_executor(
            None, _rerank_with_cross_encoder, query, documents, model_name, device
        )

    ranked_docs = sorted(scored_documents, key=lambda x: x.get("rerank_score", -1.0), reverse=True)

    return ranked_docs[:top_k]