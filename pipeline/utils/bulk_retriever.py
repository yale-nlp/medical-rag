import json
import os
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List, Set
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from xml.etree import ElementTree as ET

load_dotenv()
CONFIG_FILE = Path("configs.json")

S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY")

ALLOWED_JOURNALS = [
    "Radiographics", "Radiology", "Clinical Imaging",
    "Journal of the American College of Radiology", "Abdominal Imaging",
    "NeuroGraphics", "American Journal of Roentgenology",
    "American Journal of Neuroradiology", "European Journal of Radiology",
    "European Radiology",
]

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_CONCURRENCY_LIMIT = 1
S2_REQUEST_DELAY = 1.0

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_CONCURRENCY_LIMIT = 3
PUBMED_REQUEST_DELAY = 1

def get_all_unique_queries(keyword_file: Path) -> Set[str]:
    print(f"Extracting unique queries from {keyword_file}...")
    queries = set()
    with open(keyword_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading input file"):
            if line.strip():
                case = json.loads(line)
                queries_for_case = case.get("final_queries") or case.get("keywords", [])
                for q in queries_for_case:
                    if isinstance(q, str) and q.strip():
                        queries.add(q.strip())
    print(f"Found {len(queries)} unique queries.")
    return queries

async def rate_limited_fetch(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, delay: float, url: str, **kwargs) -> aiohttp.ClientResponse:
    async with semaphore:
        request_task = asyncio.create_task(session.get(url, **kwargs))
        delay_task = asyncio.create_task(asyncio.sleep(delay))
        await asyncio.wait([request_task, delay_task], return_when=asyncio.ALL_COMPLETED)
        return await request_task

async def fetch_s2_for_query(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, query: str) -> Dict:
    params = {"query": query, "fields": "title,url,abstract,venue,year,authors", "limit": 15}
    if ALLOWED_JOURNALS:
        params["venue"] = ",".join(ALLOWED_JOURNALS)
    headers = {"x-api-key": S2_API_KEY}
    try:
        response = await rate_limited_fetch(session, semaphore, S2_REQUEST_DELAY, S2_SEARCH_URL, params=params, headers=headers)
        response.raise_for_status()
        data = await response.json()
        results = [{"paperId": p.get("paperId"), "title": p.get("title", "N/A"), "url": p.get("url", "N/A"), "abstract": p.get("abstract", "N/A"), "source": "SemanticScholar"} for p in data.get("data", [])]
        return {"query": query, "results": results}
    except aiohttp.ClientError as e:
        tqdm.write(f"\n[ERROR] S2 search failed for query '{query}': {e}")
        return {"query": query, "results": []}

async def retrieve_s2_results_async(queries: List[str], output_file: Path):
    print("\n--- Starting Semantic Scholar Asynchronous Retrieval (Rate-Limited) ---")
    semaphore = asyncio.Semaphore(S2_CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_s2_for_query(session, semaphore, query) for query in queries]
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching S2 results"):
                result = await future
                if result and result["results"]:
                    f_out.write(json.dumps(result) + '\n')
    print(f"[SUCCESS] Semantic Scholar retrieval complete. Results saved to {output_file}")

async def fetch_pubmed_for_query(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, query: str) -> Dict:
    pmids = []
    search_term = query
    if ALLOWED_JOURNALS:
        journal_filter = " OR ".join([f'"{j}"[Journal]' for j in ALLOWED_JOURNALS])
        search_term = f'("{query}") AND ({journal_filter})'
    esearch_params = {"db": "pubmed", "term": search_term, "retmode": "json", "retmax": 15, "api_key": PUBMED_API_KEY}
    try:
        response = await rate_limited_fetch(session, semaphore, PUBMED_REQUEST_DELAY, PUBMED_ESEARCH_URL, params=esearch_params)
        response.raise_for_status()
        data = await response.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])
    except aiohttp.ClientError as e:
        tqdm.write(f"\n[ERROR] PubMed esearch failed for query '{query}': {e}")
        return {"query": query, "results": []}

    if not pmids: return {"query": query, "results": []}

    efetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "api_key": PUBMED_API_KEY}
    try:
        response = await rate_limited_fetch(session, semaphore, PUBMED_REQUEST_DELAY, PUBMED_EFETCH_URL, params=efetch_params)
        response.raise_for_status()
        xml_text = await response.text()
        root = ET.fromstring(xml_text)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID"); pmid = pmid_elem.text if pmid_elem is not None else "N/A"
            title_elem = article.find(".//ArticleTitle"); title = "".join(title_elem.itertext()).strip() if title_elem is not None else "No Title"
            abstract_parts = [f'{abst.get("Label", "")}: {"".join(abst.itertext()).strip()}' if abst.get("Label") else "".join(abst.itertext()).strip() for abst in article.findall(".//Abstract/AbstractText")]
            abstract = "\n".join(filter(None, abstract_parts)) or "No Abstract"
            articles.append({"paperId": pmid, "title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", "abstract": abstract, "source": "PubMed"})
        return {"query": query, "results": articles}
    except (aiohttp.ClientError, ET.ParseError) as e:
        tqdm.write(f"\n[ERROR] PubMed efetch failed for query '{query}': {e}")
        return {"query": query, "results": []}

async def retrieve_pubmed_results_async(queries: List[str], output_file: Path):
    print("\n--- Starting PubMed Asynchronous Retrieval (Rate-Limited) ---")
    semaphore = asyncio.Semaphore(PUBMED_CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_pubmed_for_query(session, semaphore, query) for query in queries]
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching PubMed results"):
                result = await future
                if result and result["results"]:
                    f_out.write(json.dumps(result) + '\n')
    print(f"[SUCCESS] PubMed retrieval complete. Results saved to {output_file}")

async def main():
    print("--- Starting Bulk Article Retriever (Async & Rate-Limited Version) ---")
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        keyword_file = Path(config["keyword_file"])
        
        output_dir = Path("retrieved_data")
        output_dir.mkdir(exist_ok=True)
        
        unique_queries = list(get_all_unique_queries(keyword_file))
        if not unique_queries:
            print("[INFO] No unique queries found. Exiting."); return

        main_tasks = []

        s2_output_file = output_dir / "s2_opensource.jsonl"
        if S2_API_KEY:
            if s2_output_file.exists():
                print(f"[INFO] Semantic Scholar results file already exists: {s2_output_file}. Skipping.")
            else:
                main_tasks.append(retrieve_s2_results_async(unique_queries, s2_output_file))
        else:
            print("[WARN] SEMANTIC_SCHOLAR_API_KEY not set. Skipping Semantic Scholar retrieval.")

        pubmed_output_file = output_dir / "pubmed_opensource.jsonl"
        if PUBMED_API_KEY:
            if pubmed_output_file.exists():
                print(f"[INFO] PubMed results file already exists: {pubmed_output_file}. Skipping.")
            else:
                main_tasks.append(retrieve_pubmed_results_async(unique_queries, pubmed_output_file))
        else:
            print("[WARN] PUBMED_API_KEY not set. Skipping PubMed retrieval.")
        
        if main_tasks:
            await asyncio.gather(*main_tasks)
            print("\nAll retrieval tasks have been completed.")
        else:
            print("\nNo new retrieval tasks to run.")

    except Exception as e:
        print(f"\n[FATAL] An error occurred: {e}")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")