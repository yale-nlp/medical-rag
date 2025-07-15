import os
import asyncio
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from tqdm.asyncio import tqdm
from multiprocessing import Process, Queue
import glob
from dotenv import load_dotenv

# --- Module Imports ---
import config_loader
from llm_services import LocalLLMService
from utils.retrieval import retrieve_and_rerank_for_case_async
from utils.qwen_parsing import extract_qwen_response
from utils.local_rag import load_embeddings, embed_queries
from utils.vllm_handler import VLLMHandler

load_dotenv()

# --- Concurrency Control ---
RETRIEVAL_CONCURRENCY_LIMIT = 200
GENERATION_CONCURRENCY_LIMIT = 200
API_REQUEST_CONCURRENCY_LIMIT = 3

class ModelWorker(Process):
    def __init__(self, config, gpu_id, request_queue, response_queue):
        super().__init__()
        self.config = config
        self.gpu_id = gpu_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.handler = None

    def setup_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        from vllm import LLM
        from transformers import AutoTokenizer
        
        logging.info(f"[Worker-{self.gpu_id}] Loading model '{self.config['model']}' onto GPU(s) {self.gpu_id}...")
        if logging.getLogger().getEffectiveLevel() > logging.DEBUG:
            logging.getLogger("vllm").setLevel(logging.WARNING)

        model = LLM(
            model=self.config['model'],
            tensor_parallel_size=self.config.get("tensor_parallel_size", 1),
            max_model_len=self.config.get("max_model_len", 8192),
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
            quantization=self.config.get("quantization"),
            max_num_seqs=200
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config['model'], trust_remote_code=True)
        self.handler = VLLMHandler(model, tokenizer, self.config)
        logging.info(f"[Worker-{self.gpu_id}] Model '{self.config['model']}' and handler loaded successfully.")

    def cleanup(self):
        """Explicitly cleans up GPU resources before the process exits."""
        logging.info(f"[Worker-{self.gpu_id}] Cleaning up GPU resources...")
        try:
            if hasattr(self, 'handler') and self.handler:
                if hasattr(self.handler, 'model'):
                    del self.handler.model
                del self.handler
            
            import torch
            torch.cuda.empty_cache()
            logging.info(f"[Worker-{self.gpu_id}] GPU resources cleaned up successfully.")
        except Exception as e:
            logging.error(f"[Worker-{self.gpu_id}] Error during cleanup: {e}", exc_info=True)

    def run(self):
        try:
            self.setup_model()
            while True:
                request = self.request_queue.get()
                if request is None:
                    logging.debug(f"[Worker-{self.gpu_id}] Received shutdown signal.")
                    self.cleanup()
                    break
                
                request_id, payload = request["request_id"], request["payload"]
                task = payload["task"]

                if task == "generate_batch":
                    result = self.handler.generate_text_batch(payload)
                elif task == "rerank":
                    result = self.handler.rerank(payload)
                else:
                    result = {"error": f"Unknown task: {task}"}
                
                self.response_queue.put({"request_id": request_id, "result": result})
        except Exception as e:
            logging.critical(f"[Worker-{self.gpu_id}] CRASHED: {e}", exc_info=True)
            self.response_queue.put({"request_id": "CRITICAL_ERROR", "result": {"error": str(e)}})

async def result_listener(response_queue: Queue, pending_futures: dict):
    while True:
        if not response_queue.empty():
            response = response_queue.get()
            request_id = response.get("request_id")
            if request_id == "CRITICAL_ERROR":
                logging.critical(f"A worker process has crashed: {response.get('result')}")
                continue
            future = pending_futures.get(request_id)
            if future and not future.done():
                result_payload = response.get("result")
                if isinstance(result_payload, dict) and "error" in result_payload:
                    future.set_exception(RuntimeError(f"Worker task failed: {result_payload['error']}"))
                else:
                    future.set_result(result_payload)
        await asyncio.sleep(0.01)

def start_model_workers(config: dict) -> Dict:
    workers, comms = {}, {}
    service_configs = config.get("services", {})
    for service_name, settings in service_configs.items():
        gpu_id = settings.get("gpu_id")
        if gpu_id is None:
            logging.warning(f"No 'gpu_id' for service '{service_name}'. Skipping worker.")
            continue
        req_q, res_q, pending_futures_dict = Queue(), Queue(), {}
        worker = ModelWorker(settings, str(gpu_id), req_q, res_q)
        worker.start()
        workers[service_name] = worker
        comms[service_name] = {
            "request_queue": req_q,
            "response_queue": res_q,
            "pending_futures_dict": pending_futures_dict,
            "model_name": settings["model"]
        }
    return workers, comms

def load_prefetched_articles(path: Path) -> Dict[str, List[Dict]]:
    if not path.exists():
        return {}
    articles_by_query = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if "query" in data and "results" in data:
                    articles_by_query[data["query"]] = data["results"]
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed line in {path}: {line.strip()}")
    return articles_by_query

# --- MODIFICATION: Added case_id_filter parameter for UI integration ---
async def main(config: dict, case_id_filter: str = None):
    """Main pipeline execution logic."""
    logging.info("--- Starting Asynchronous Medical RAG Pipeline (Batch-Stage Mode) ---")

    keyword_file = Path(config["keyword_file"])
    if not keyword_file.exists():
        logging.fatal(f"Keyword file not found at: {keyword_file}")
        return

    feedback_dir = Path(config["feedback_dir"])
    feedback_dir.mkdir(parents=True, exist_ok=True)

    workers, comms, listener_tasks = {}, {}, []
    try:
        workers, comms = start_model_workers(config)
        
        service_map = config.get("service_map", {})
        if not service_map:
            raise ValueError("Config error: 'service_map' is missing or empty.")
        service_clients = {s_name: LocalLLMService(model_name=c["model_name"], request_queue=c["request_queue"], pending_futures_dict=c["pending_futures_dict"]) for s_name, c in comms.items()}
        llm_services = {stage_name: service_clients[service_name_from_map] for stage_name, service_name_from_map in service_map.items()}

        listener_tasks = [asyncio.create_task(result_listener(c["response_queue"], c["pending_futures_dict"])) for c in comms.values()]
        
        logging.info("Loading data and computing CPU embeddings...")
        all_case_data = [json.loads(line) for line in keyword_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        # --- MODIFICATION: Filter for a single case if case_id_filter is provided by the UI ---
        if case_id_filter:
            logging.info(f"Filtering pipeline run for single Case ID: {case_id_filter}")
            all_case_data = [case for case in all_case_data if case.get("case_id") == case_id_filter]
            if not all_case_data:
                logging.fatal(f"Case ID '{case_id_filter}' not found in keyword file: {keyword_file}")
                return

        for case in all_case_data:
            case["original_keywords"] = case.get("keywords", [])

        pages_with_embeddings = load_embeddings(config["local_index_path"]) if "local" in config.get("retrievers", []) else []
        kw_embeddings_map = {}
        if pages_with_embeddings:
            all_kws = list(set(kw.strip() for case in all_case_data for kw in case.get("original_keywords", []) if isinstance(kw, str) and kw.strip()))
            if all_kws:
                kw_embeddings_map = dict(zip(all_kws, embed_queries(all_kws, config["hf_embedding_model"], config["embedding_model_device"])))
        
        prefetched_articles = {}
        s2_path_str = config.get("s2_retrieved_file")
        pubmed_path_str = config.get("pubmed_retrieved_file")
        if s2_path_str:
            s2_path = Path(s2_path_str)
            if s2_path.is_file():
                prefetched_articles.update(load_prefetched_articles(s2_path))
            else:
                logging.warning(f"Path for 's2_retrieved_file' provided but is not a valid file: {s2_path}")
        if pubmed_path_str:
            pubmed_path = Path(pubmed_path_str)
            if pubmed_path.is_file():
                prefetched_articles.update(load_prefetched_articles(pubmed_path))
            else:
                logging.warning(f"Path for 'pubmed_retrieved_file' provided but is not a valid file: {pubmed_path}")
        logging.info(f"Data loading complete. Processing {len(all_case_data)} cases.")

        logging.info("--- Stage 1: Retrieval & Reranking ---")
        api_semaphore = asyncio.Semaphore(API_REQUEST_CONCURRENCY_LIMIT)
        retrieval_tasks = [
            retrieve_and_rerank_for_case_async(
                case, llm_services, config, pages_with_embeddings, 
                kw_embeddings_map, prefetched_articles, api_semaphore
            ) for case in all_case_data
        ]
        retrieved_contexts = await tqdm.gather(*retrieval_tasks, desc="Retrieving & Reranking")
        
        for i, case in enumerate(all_case_data):
            case.update(retrieved_contexts[i])

        logging.info("--- Stage 2: Generating Textbook Summaries ---")
        generation_config = config.get("services", {}).get("generation_service", {})
        summary_prompts_cfg = config.get("system", {}).get("textbook_summary", {})
        summary_settings = generation_config.get("generation_params", {}).get("textbook_summary", {})
        
        summary_prompts_to_generate = []
        summary_case_map = [] 

        for i, case in enumerate(all_case_data):
            case["generated_textbook_summaries"] = {}
            for kw, pages in case.get("local_pages_per_keyword", {}).items():
                if pages and isinstance(pages, list):
                    pages_block = "\n\n".join([f"Page {p.get('index', 'N/A')}:\n{p.get('text', '')}" for p in pages])
                    if pages_block.strip():
                        user_prompt = summary_prompts_cfg["user_instruction_template"].format(keyword=kw, pages_block_text=pages_block)
                        summary_prompts_to_generate.append((summary_prompts_cfg["system_persona"], user_prompt))
                        summary_case_map.append({'case_idx': i, 'keyword': kw})

        if summary_prompts_to_generate:
            raw_summaries = await llm_services["textbook_summary"].generate_text_batch(summary_prompts_to_generate, summary_settings)
            is_qwen_model = "qwen" in generation_config.get("model", "").lower()
            all_summaries = [extract_qwen_response(res) for res in raw_summaries] if is_qwen_model else raw_summaries

            for idx, mapping_info in enumerate(summary_case_map):
                all_case_data[mapping_info['case_idx']]["generated_textbook_summaries"][mapping_info['keyword']] = all_summaries[idx]
        
        logging.info("--- Stage 3: Generating Final Reports & MCQs ---")
        final_report_prompts = []
        mcq_prompts = []
        
        report_cfg = config.get("system", {}).get("final_feedback", {})
        mcq_cfg = config.get("system", {}).get("questioner", {})
        report_settings = generation_config.get("generation_params", {}).get("final_report", {})
        mcq_settings = generation_config.get("generation_params", {}).get("mcq_generation", {})

        for case in all_case_data:
            original_keywords = case.get("original_keywords", [])
            original_reviewer_report = case.get("reviewer_report", "")
            summaries = case.get("generated_textbook_summaries", {})
            user_block = ""
            reranked_papers_data = case.get("reranked_papers_per_keyword", {})
            
            for kw in original_keywords:
                user_block += f"### Keyword: {kw}\n**Reranked Paper Abstracts:**\n"
                reranked_docs = reranked_papers_data.get(kw, [])
                if reranked_docs:
                    for i, doc in enumerate(reranked_docs, 1):
                        score, title, abstract, url, source = (doc.get('rerank_score', 0.0), doc.get('title', 'N/A'), doc.get('abstract', 'N/A'), doc.get('url', 'N/A'), doc.get('source', 'N/A'))
                        user_block += (f"{i}. **Title:** {title} (Score: {score:.4f})\n   **Abstract:** {abstract}\n   **URL:** {url}\n   **Source:** {source}\n\n")
                else:
                    user_block += "N/A\n\n"
                user_block += "**Textbook Summary:**\n" + summaries.get(kw, "N/A") + "\n\n"
            
            user_block += "---\n"
            final_report_prompt = report_cfg["user_instruction_template"].format(keywords_list_str="\n- ".join(original_keywords), original_reviewer_report=original_reviewer_report, user_block_for_final_stages=user_block)
            final_report_prompts.append((report_cfg["system_persona"], final_report_prompt))
            
            mcq_input_context = (f"Original Radiology Report (for context):\n{original_reviewer_report}\n\n--- Educational Material for Keywords ---\n{user_block}")
            mcq_prompt = mcq_cfg["user_instruction_template"].format(keywords_list_str="\n- ".join(original_keywords), mcq_input_context=mcq_input_context)
            mcq_prompts.append((mcq_cfg["system_persona"], mcq_prompt))

        report_batch_task = llm_services["final_report"].generate_text_batch(final_report_prompts, report_settings)
        mcq_batch_task = llm_services["mcq_generation"].generate_text_batch(mcq_prompts, mcq_settings)
        raw_final_reports, raw_mcqs = await asyncio.gather(report_batch_task, mcq_batch_task)
        
        is_qwen_model_gen = "qwen" in generation_config.get("model", "").lower()
        final_feedbacks = [extract_qwen_response(res) for res in raw_final_reports] if is_qwen_model_gen else raw_final_reports
        all_mcqs = [extract_qwen_response(res) for res in raw_mcqs] if is_qwen_model_gen else raw_mcqs
        
        all_reports_output = []
        for i, case in enumerate(all_case_data):
            final_structured_output = {
                "case_id": case.get("case_id"),
                "original_keywords": case.get("original_keywords"),
                "original_reviewer_report": case.get("reviewer_report"),
                "evidence_reranked_papers": case.get("reranked_papers_per_keyword", {}),
                "evidence_retrieved_textbook_pages": case.get("local_pages_per_keyword", {}),
                "generated_textbook_summaries": case.get("generated_textbook_summaries", {}),
                "generated_final_feedback": final_feedbacks[i],
                "generated_mcqs": all_mcqs[i]
            }
            all_reports_output.append(final_structured_output)

        output_path = feedback_dir / (keyword_file.stem + "_local_vllm_feedback.json")
        final_output = {
            "pipeline_run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "pipeline_configuration": {k: v for k, v in config.items() if not k.startswith("_")},
            "all_processed_reports": all_reports_output
        }
        output_path.write_text(json.dumps(final_output, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info(f"Pipeline finished. All {len(all_reports_output)} processed cases saved to: {output_path}")

    finally:
        logging.info("Shutting down model workers...")
        for task in listener_tasks:
            task.cancel()
        for comm_data in comms.values():
            comm_data["request_queue"].put(None)
        
        for worker in workers.values():
            worker.join(timeout=30)
            if worker.is_alive():
                logging.warning(f"Worker {worker.pid} did not exit gracefully. Terminating.")
                worker.terminate()
                worker.join()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    # 1. Set up the command-line argument parser.
    parser = argparse.ArgumentParser(description="Run the Asynchronous Medical RAG Pipeline with Local vLLM.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs.json",
        help="Path to the configuration JSON file."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed logs."
    )
    # --- MODIFICATION: Added case_id argument for UI control ---
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Run the pipeline for a single case_id. If not provided, runs for all cases."
    )
    args = parser.parse_args()

    # 2. Set up logging to both console and file for the UI.
    log_file_path = 'gradio/pipeline.log'
    # --- MODIFICATION: Clear previous log file for a clean run ---
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    config_loader.setup_logging(args.debug)
    
    # --- MODIFICATION: Add a file handler to log to pipeline.log for the UI ---
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    log_level = logging.DEBUG if args.debug else logging.INFO
    file_handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] - %(message)s', datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    # 3. Set the multiprocessing start method.
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 4. Load the configuration file specified by the argument.
    try:
        logging.info(f"Loading configuration from: {args.config}")
        config = config_loader.load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logging.fatal(f"Could not load configuration: {e}")
        exit(1)

    # 5. Execute the main pipeline.
    start_time = time.time()
    # --- MODIFICATION: Pass the case_id from args to the main function ---
    asyncio.run(main(config, case_id_filter=args.case_id))
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")