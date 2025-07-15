import os
import asyncio
import logging
import time
import requests
from xml.etree import ElementTree as ET
from typing import List, Dict
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

ALLOWED_JOURNALS = [
    "Radiographics", 
    "Radiology", 
    "Clinical Imaging",
    "Journal of the American College of Radiology", 
    "Abdominal Imaging",
    "NeuroGraphics", 
    "American Journal of Roentgenology",
    "American Journal of Neuroradiology", 
    "European Journal of Radiology",
    "European Radiology",
]

def requests_retry_session(retries=3, backoff_factor=0.5, session=None):
    """A factory for requests sessions with built-in retry mechanism."""
    session = session or requests.Session()
    retry = requests.packages.urllib3.util.retry.Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504, 429), # Retry on server errors and rate limits
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

async def run_sync_in_thread(func, *args, **kwargs):
    """Runs a synchronous function in a separate thread to avoid blocking asyncio loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def search_semantic_scholar(keyword: str, semaphore: asyncio.Semaphore, limit: int = 5) -> List[Dict]:
    """Performs a live search on Semantic Scholar API with semaphore and retry."""
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        logging.warning("SEMANTIC_SCHOLAR_API_KEY not set. Skipping Semantic Scholar search.")
        return []

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    headers = {"x-api-key": api_key}
    params = {"query": keyword, "fields": "title,venue,year,url,abstract", "limit": limit}
    if ALLOWED_JOURNALS:
        params["venue"] = ",".join(ALLOWED_JOURNALS)

    async with semaphore:
        try:
            response = await run_sync_in_thread(
                requests_retry_session().get,
                url=base_url, headers=headers, params=params, timeout=15
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            return [{"paperId": p.get("paperId"), "title": p.get("title", "N/A"), "url": p.get("url", "N/A"), "abstract": p.get("abstract", "N/A"), "source": "SemanticScholar"} for p in data]
        except Exception as e:
            logging.error(f"Semantic Scholar API request failed after retries (keyword: {keyword}): {e}")
            return []

async def search_and_fetch_pubmed(keyword: str, semaphore: asyncio.Semaphore, retmax: int = 5) -> List[Dict]:
    """Performs a live search on PubMed API with semaphore and retry."""
    search_term = f"({keyword}) AND ({' OR '.join([f'{j}[Journal]' for j in ALLOWED_JOURNALS])})" if ALLOWED_JOURNALS else keyword
    api_key = os.getenv("PUBMED_API_KEY")
    session = requests_retry_session()

    async with semaphore:
        try:
            search_params = {"db": "pubmed", "term": search_term, "retmode": "json", "retmax": retmax}
            if api_key: search_params["api_key"] = api_key
            res = await run_sync_in_thread(session.get, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=search_params, timeout=15)
            res.raise_for_status()
            pmid_list = res.json().get("esearchresult", {}).get("idlist", [])
            if not pmid_list: return []
        except Exception as e:
            logging.error(f"PubMed ESearch API request failed after retries (keyword: {keyword}): {e}")
            return []

    await asyncio.sleep(0.1) # Small delay between esearch and efetch

    async with semaphore:
        try:
            fetch_params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
            if api_key: fetch_params["api_key"] = api_key
            res = await run_sync_in_thread(session.get, "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params, timeout=15)
            res.raise_for_status()
            root = ET.fromstring(res.content)
            papers = []
            for article in root.findall(".//PubmedArticle"):
                paper_id = (article.find(".//PMID") or ET.Element("")).text or "N/A"
                title = "".join((article.find(".//ArticleTitle") or ET.Element("")).itertext()).strip() or "N/A"
                abstract_parts = ["".join(abst.itertext()).strip() for abst in article.findall(".//Abstract/AbstractText")]
                papers.append({"paperId": paper_id, "title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{paper_id}/", "abstract": "\n".join(filter(None, abstract_parts)) or "No Abstract", "source": "PubMed"})
            return papers
        except Exception as e:
            logging.error(f"PubMed EFetch API request failed after retries (IDs: {pmid_list}): {e}")
            return []

async def search_live_apis_async(keyword: str, semaphore: asyncio.Semaphore) -> List[Dict]:
    """Concurrently calls S2 and PubMed APIs under a shared semaphore and merges the results."""
    logging.debug(f"Performing live search for: '{keyword}'...")
    
    s2_task = search_semantic_scholar(keyword, semaphore)
    pubmed_task = search_and_fetch_pubmed(keyword, semaphore)
    
    results = await asyncio.gather(s2_task, pubmed_task)
    
    combined_papers = [paper for res_list in results for paper in res_list]
        
    seen_ids = set()
    unique_papers = []
    for paper in combined_papers:
        uid = paper.get("paperId")
        if uid and uid not in seen_ids:
            unique_papers.append(paper)
            seen_ids.add(uid)
            
    logging.debug(f"Live search for '{keyword}' complete. Found {len(unique_papers)} unique papers.")
    return unique_papers