import asyncio
from typing import Dict, Any, List
import logging
from llm_services import LocalLLMService
from utils.reranker import rerank_candidates_async
from utils.local_rag import retrieve_top_index, get_surrounding_pages
from utils.live_retriever import search_live_apis_async

async def retrieve_and_rerank_for_case_async(
    case_data: Dict[str, Any],
    llm_services: Dict[str, LocalLLMService],
    config: Dict[str, Any],
    pages_with_embeddings: List[Dict[str, Any]],
    kw_embeddings_map: Dict[str, Any],
    prefetched_articles: Dict[str, List[Dict]],
    api_semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """
    Performs retrieval and reranking using a hybrid approach,
    with an added semaphore to limit concurrent API calls.
    """
    original_keywords = case_data.get("original_keywords", [])
    retrievers = config.get("retrievers", [])
    
    reranked_papers_per_keyword = {}
    
    if ("semantic" in retrievers or "pubmed" in retrievers):
        # --- ### 수정된 부분: 새로운 config 구조 경로 사용 ### ---
        reranker_settings = config.get("services", {}).get("reranker_service", {})
        # --- ### 수정 끝 ### ---
        
        qwen_params = reranker_settings.get("qwen_reranker_params", {})
        reranker_service = llm_services.get("reranker")

        rerank_tasks = []
        keywords_for_tasks = []

        for keyword in original_keywords:
            papers_for_keyword = prefetched_articles.get(keyword)
            
            if papers_for_keyword is None:
                papers_for_keyword = await search_live_apis_async(keyword, api_semaphore)
            else:
                logging.debug(f"Using prefetched data for '{keyword}'.")

            if not papers_for_keyword:
                logging.debug(f"No articles found for keyword: '{keyword}'. Skipping reranking.")
                reranked_papers_per_keyword[keyword] = []
                continue

            contextual_query = f"Report Context: {case_data.get('reviewer_report', '')}\n\nKeyword to focus on: '{keyword}'"
            
            task = rerank_candidates_async(
                query=contextual_query, 
                documents=papers_for_keyword,
                top_k=reranker_settings.get("rerank_candidates_num", 2),
                model_name=reranker_settings.get("model"), # 이제 None이 아님
                device=config.get("embedding_model_device", "cpu"),
                instruction_task=qwen_params.get("qwen_instruction_task", ""),
                llm_service=reranker_service
            )
            rerank_tasks.append(task)
            keywords_for_tasks.append(keyword)
        
        if rerank_tasks:
            reranked_results = await asyncio.gather(*rerank_tasks)
            for i, keyword in enumerate(keywords_for_tasks):
                reranked_papers_per_keyword[keyword] = reranked_results[i]

    local_pages_per_keyword = {}
    if "local" in retrievers and pages_with_embeddings and original_keywords:
        for kw in original_keywords:
            kw_emb = kw_embeddings_map.get(kw)
            if kw_emb is not None and kw_emb.size > 0:
                top_idx = retrieve_top_index(kw_emb, pages_with_embeddings)
                local_pages_per_keyword[kw] = get_surrounding_pages(top_idx, pages_with_embeddings)

    retrieved_count = sum(len(v) for v in reranked_papers_per_keyword.values())

    return {
        "reranked_papers_per_keyword": reranked_papers_per_keyword,
        "local_pages_per_keyword": local_pages_per_keyword,
        "debug_info": {"retrieved_and_reranked_papers_count": retrieved_count}
    }