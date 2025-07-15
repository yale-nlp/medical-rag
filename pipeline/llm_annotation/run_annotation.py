import json
import re
import os
import argparse
import asyncio
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
from threading import Thread

import numpy as np
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm

from annotator import get_annotator, BaseAnnotator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AnnotationTask:
    """Represents a single annotation task"""
    case_id: str
    task_type: str
    item_key: str
    prompt: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationResult:
    """Represents the result of an annotation task"""
    task: AnnotationTask
    score: Optional[int] = None
    error: Optional[str] = None
    duration: float = 0.0


class WorkerPool:
    """Professional worker pool for processing annotation tasks"""
    
    def __init__(self, annotator: BaseAnnotator, num_workers: int = 3):
        self.annotator = annotator
        self.num_workers = num_workers
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.result_queue: asyncio.Queue = asyncio.Queue()
        self.workers: List[asyncio.Task] = []
        self.stop_event = asyncio.Event()
        self.stats = {
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "total_duration": 0.0
        }
    
    async def start(self):
        """Start the worker pool"""
        logger.info(f"Starting worker pool with {self.num_workers} workers")
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(f"Worker-{i}"))
            self.workers.append(worker)
    
    async def stop(self):
        """Stop the worker pool gracefully"""
        logger.info("Stopping worker pool...")
        self.stop_event.set()
        
        # Wait for all workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Worker pool stopped")
    
    async def _worker(self, worker_name: str):
        """Worker coroutine that processes tasks"""
        logger.debug(f"{worker_name} started")
        
        while not self.stop_event.is_set():
            try:
                # Get task with timeout to check stop event periodically
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Process the task
                start_time = time.time()
                try:
                    if hasattr(self.annotator, 'evaluate_async'):
                        result_dict = await self.annotator.evaluate_async(task.prompt)
                    else:
                        # Run sync method in thread pool
                        loop = asyncio.get_event_loop()
                        result_dict = await loop.run_in_executor(
                            None, self.annotator.evaluate, task.prompt
                        )
                    
                    duration = time.time() - start_time
                    
                    # Create result
                    if "error" in result_dict:
                        result = AnnotationResult(
                            task=task,
                            error=result_dict["error"],
                            duration=duration
                        )
                        self.stats["failed"] += 1
                    else:
                        result = AnnotationResult(
                            task=task,
                            score=self._to_int_score(result_dict.get("score")),
                            duration=duration
                        )
                        self.stats["succeeded"] += 1
                    
                    self.stats["processed"] += 1
                    self.stats["total_duration"] += duration
                    
                    # Put result in queue
                    await self.result_queue.put(result)
                    
                except Exception as e:
                    logger.error(f"{worker_name} error processing task: {e}")
                    result = AnnotationResult(
                        task=task,
                        error=str(e),
                        duration=time.time() - start_time
                    )
                    await self.result_queue.put(result)
                    self.stats["failed"] += 1
                    self.stats["processed"] += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"{worker_name} unexpected error: {e}")
        
        logger.debug(f"{worker_name} stopped")
    
    @staticmethod
    def _to_int_score(value) -> Optional[int]:
        """Convert value to integer score"""
        if value is None:
            return None
        try:
            score = int(value)
            return score if 1 <= score <= 5 else None
        except (ValueError, TypeError):
            return None
    
    async def submit_task(self, task: AnnotationTask):
        """Submit a task to the pool"""
        await self.task_queue.put(task)
    
    async def get_result(self) -> Optional[AnnotationResult]:
        """Get a result from the pool"""
        try:
            return await asyncio.wait_for(self.result_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self.stats.copy()
        if stats["processed"] > 0:
            stats["average_duration"] = stats["total_duration"] / stats["processed"]
            stats["success_rate"] = stats["succeeded"] / stats["processed"]
        return stats


class AnnotationPipeline:
    """Main annotation pipeline using producer-consumer pattern"""
    
    def __init__(self, annotator: BaseAnnotator, num_workers: int = 3):
        self.annotator = annotator
        self.worker_pool = WorkerPool(annotator, num_workers)
        self.progress_bar = None
    
    async def process_cases(self, cases: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process all cases using the worker pool"""
        # Generate all tasks
        all_tasks = self._generate_all_tasks(cases)
        total_tasks = len(all_tasks)
        
        logger.info(f"Processing {total_tasks} tasks across {len(cases)} cases")
        
        # Start worker pool
        await self.worker_pool.start()
        
        # Progress tracking
        if show_progress:
            self.progress_bar = tqdm(total=total_tasks, desc="Processing annotations")
        
        try:
            # Submit all tasks
            submit_task = asyncio.create_task(self._submit_all_tasks(all_tasks))
            
            # Collect results
            collect_task = asyncio.create_task(self._collect_results(cases, total_tasks))
            
            # Wait for both to complete
            await asyncio.gather(submit_task, collect_task)
            
        finally:
            # Stop worker pool
            await self.worker_pool.stop()
            
            if self.progress_bar:
                self.progress_bar.close()
        
        # Log final statistics
        stats = self.worker_pool.get_stats()
        logger.info(f"Processing complete: {stats}")
        
        return cases
    
    def _generate_all_tasks(self, cases: List[Dict[str, Any]]) -> List[AnnotationTask]:
        """Generate all annotation tasks from cases"""
        tasks = []
        
        for case in cases:
            case_id = case.get("case_id")
            if not case_id:
                continue
            
            prompts = self._generate_prompts_for_case(case)
            
            for prompt_id, prompt_text in prompts:
                parts = prompt_id.split("__")
                if len(parts) >= 3:
                    task = AnnotationTask(
                        case_id=parts[0],
                        task_type=parts[1],
                        item_key=parts[2],
                        prompt=prompt_text
                    )
                    tasks.append(task)
        
        return tasks
    
    def _generate_prompts_for_case(self, case_data: dict) -> List[Tuple[str, str]]:
        """Generate prompts for a single case"""
        case_id = case_data.get("case_id")
        original_report = case_data.get("original_reviewer_report")
        original_keywords = case_data.get("original_keywords", [])
        
        if not all([case_id, original_report, original_keywords]):
            return []
        
        prompts = []
        paper_fallback_counter = 0
        
        # Keyword appropriateness
        for kw in original_keywords:
            kw_context = {"original_report": original_report, "keyword": kw}
            prompts.append((
                f"{case_id}__keyword_appropriateness__{kw}",
                self.annotator._construct_prompt("keyword_appropriateness_evaluation", kw_context)
            ))
            
            # Paper relevance
            for paper in case_data.get("evidence_reranked_papers", {}).get(kw, []):
                paper_id = paper.get("url") or f"no_url_paper_{paper_fallback_counter}"
                paper_fallback_counter += 1 if not paper.get("url") else 0
                
                paper_context = {
                    **kw_context,
                    "current_paper_title": paper.get("title", "N/A"),
                    "current_paper_abstract": paper.get("abstract", "N/A")
                }
                prompts.append((
                    f"{case_id}__paper_relevance__{paper_id}",
                    self.annotator._construct_prompt("reranked_results_evaluation", paper_context)
                ))
            
            # Textbook summary
            summary = case_data.get("generated_textbook_summaries", {}).get(kw)
            if summary:
                prompts.append((
                    f"{case_id}__textbook_summary_quality__{kw}",
                    self.annotator._construct_prompt("textbook_summary_evaluation", {
                        **kw_context,
                        "textbook_summary_text": summary
                    })
                ))
        
        # MCQ evaluation
        mcqs_text = case_data.get("generated_mcqs", "")
        if mcqs_text:
            for i, match in enumerate(re.finditer(
                r"Q\d+\.\s*(?P<q>[\s\S]+?)\nAnswer:\s*(?P<a>[^\n]+)\nExplanation:\s*(?P<e>[\s\S]+?)(?=\nQ\d+|$)",
                mcqs_text,
                re.I
            )):
                mcq_context = {
                    "keyword": "overall case",
                    "mcq_question_text": match.group('q').strip(),
                    "mcq_answer_text": match.group('a').strip(),
                    "mcq_explanation_text": match.group('e').strip()
                }
                prompts.append((
                    f"{case_id}__mcq_quality__MCQ_{i+1}",
                    self.annotator._construct_prompt("mcq_evaluation", mcq_context)
                ))
        
        # Final feedback
        feedback_text = case_data.get("generated_final_feedback", "")
        if feedback_text:
            feedback_context = {
                "original_report": original_report,
                "keywords_list_str": "\n- ".join(original_keywords),
                "full_generated_feedback": feedback_text
            }
            prompts.append((
                f"{case_id}__final_feedback_quality__-",
                self.annotator._construct_prompt("final_feedback_synthesis_evaluation", feedback_context)
            ))
        
        # Overall quality
        all_papers_text = "\n\n".join([
            f"Title: {p.get('title', 'N/A')}\nAbstract: {p.get('abstract', 'N/A')}"
            for kw_papers in case_data.get("evidence_reranked_papers", {}).values()
            for p in kw_papers
        ])
        
        overall_context = {
            "original_report": original_report,
            "keywords_list_str": "\n- ".join(original_keywords),
            "all_reranked_papers_text": all_papers_text or "N/A",
            "all_mcqs_text": mcqs_text or "N/A",
            "full_generated_feedback": feedback_text or "N/A"
        }
        prompts.append((
            f"{case_id}__overall_quality__-",
            self.annotator._construct_prompt("overall_case_evaluation", overall_context)
        ))
        
        return prompts
    
    async def _submit_all_tasks(self, tasks: List[AnnotationTask]):
        """Submit all tasks to the worker pool"""
        for task in tasks:
            await self.worker_pool.submit_task(task)
        
        logger.info(f"All {len(tasks)} tasks submitted")
    
    async def _collect_results(self, cases: List[Dict[str, Any]], total_tasks: int):
        """Collect results from the worker pool"""
        case_map = {case['case_id']: case for case in cases}
        collected = 0
        
        while collected < total_tasks:
            result = await self.worker_pool.get_result()
            if result:
                # Map result back to case
                case = case_map.get(result.task.case_id)
                if case:
                    if "annotation" not in case:
                        case["annotation"] = {}
                    
                    # Store the result
                    if result.task.task_type in ["final_feedback_quality", "overall_quality"]:
                        case["annotation"][result.task.task_type] = result.score
                    else:
                        if result.task.task_type not in case["annotation"]:
                            case["annotation"][result.task.task_type] = {}
                        case["annotation"][result.task.task_type][result.task.item_key] = result.score
                
                collected += 1
                if self.progress_bar:
                    self.progress_bar.update(1)
            else:
                await asyncio.sleep(0.01)
        
        logger.info(f"All {collected} results collected")


def calculate_statistics(all_annotated_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics from annotated cases"""
    logger.info("Calculating statistics...")
    
    stats = {
        "overall_summary": {
            "total_cases_processed": len(all_annotated_cases),
            "cases_with_successful_annotation": 0
        }
    }
    
    scores = {
        "keyword_appropriateness": [],
        "paper_relevance": [],
        "textbook_summary_quality": [],
        "mcq_quality": [],
        "final_feedback_quality": [],
        "overall_quality": []
    }
    
    correlation_data = {
        "rerank_scores": [],
        "paper_relevance_scores": []
    }
    
    def to_int_score(value) -> Optional[int]:
        if value is None:
            return None
        try:
            score = int(value)
            return score if 1 <= score <= 5 else None
        except (ValueError, TypeError):
            return None
    
    for case in all_annotated_cases:
        annotation = case.get("annotation")
        if not annotation or "error" in annotation:
            continue
        
        stats["overall_summary"]["cases_with_successful_annotation"] += 1
        
        # Collect scores
        for score in annotation.get("keyword_appropriateness", {}).values():
            if (s := to_int_score(score)) is not None:
                scores["keyword_appropriateness"].append(s)
        
        for score in annotation.get("paper_relevance", {}).values():
            if (s := to_int_score(score)) is not None:
                scores["paper_relevance"].append(s)
        
        for score in annotation.get("textbook_summary_quality", {}).values():
            if (s := to_int_score(score)) is not None:
                scores["textbook_summary_quality"].append(s)
        
        for score in annotation.get("mcq_quality", {}).values():
            if (s := to_int_score(score)) is not None:
                scores["mcq_quality"].append(s)
        
        if (s := to_int_score(annotation.get("final_feedback_quality"))) is not None:
            scores["final_feedback_quality"].append(s)
        
        if (s := to_int_score(annotation.get("overall_quality"))) is not None:
            scores["overall_quality"].append(s)
        
        # Correlation data
        for kw_papers in case.get("evidence_reranked_papers", {}).values():
            for paper in kw_papers:
                relevance_score = to_int_score(
                    annotation.get("paper_relevance", {}).get(paper.get("url"))
                )
                if relevance_score is not None and paper.get("rerank_score") is not None:
                    correlation_data["rerank_scores"].append(paper.get("rerank_score"))
                    correlation_data["paper_relevance_scores"].append(relevance_score)
    
    # Calculate statistics for each category
    for category, score_list in scores.items():
        if not score_list:
            stats[category + "_stats"] = "No valid scores available."
            continue
        
        stats[category + "_stats"] = {
            "count": len(score_list),
            "mean": float(np.mean(score_list)),
            "std_dev": float(np.std(score_list)),
            "min": int(np.min(score_list)),
            "max": int(np.max(score_list)),
            "distribution": dict(sorted(Counter(score_list).items()))
        }
    
    # Calculate correlation if enough data
    if len(correlation_data["rerank_scores"]) > 1:
        correlation_matrix = np.corrcoef(
            correlation_data["rerank_scores"],
            correlation_data["paper_relevance_scores"]
        )
        stats.setdefault("paper_relevance_stats", {})["correlation_with_rerank_score"] = {
            "pearson_coefficient": float(correlation_matrix[0, 1]),
            "data_points_count": len(correlation_data["rerank_scores"])
        }
    
    return stats


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluations.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--config_file", type=str, default="annotation_configs.json", help="Configuration file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases to process")
    parser.add_argument("--workers", type=int, default=3, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    with open(args.config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    annotator_choice = config.get("annotator_choice", "openai").lower()
    
    # Set CUDA devices for HuggingFace
    if annotator_choice == "huggingface":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.get("cuda_visible_devices", "")
    
    # Get annotator
    annotator = get_annotator(args.config_file)
    model_name = config.get(f"{annotator_choice}_annotator_settings", {}).get(
        "model_name",
        config.get(f"{annotator_choice}_annotator_settings", {}).get("model")
    )
    
    # Load input data
    logger.info(f"Loading input data from {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)
    
    all_cases = input_data.get("all_processed_reports", [])
    if args.limit:
        all_cases = all_cases[:args.limit]
    
    logger.info(f"Processing {len(all_cases)} cases with {model_name}")
    
    # Create pipeline
    pipeline = AnnotationPipeline(annotator, num_workers=args.workers)
    
    # Process cases
    start_time = time.time()
    
    try:
        all_cases = await pipeline.process_cases(all_cases)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    
    elapsed_time = time.time() - start_time
    
    # Calculate statistics
    statistics_summary = calculate_statistics(all_cases)
    
    # Save results
    output_filename = f"{Path(args.input_file).stem}_annotated_by_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name)}.json"
    
    final_output_data = {
        "statistics_summary": statistics_summary,
        "pipeline_configuration": input_data.get("pipeline_configuration", {}),
        "all_processed_reports": all_cases,
        "annotation_metadata": {
            "annotator": annotator_choice,
            "model": model_name,
            "processing_time_minutes": round(elapsed_time / 60, 2),
            "workers_used": args.workers,
            "cases_processed": len(all_cases)
        }
    }
    
    output_path = output_dir / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Processing completed in {elapsed_time/60:.1f} minutes")
    
    # Print summary
    print("\n" + "="*50)
    print("ANNOTATION SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Total cases: {len(all_cases)}")
    print(f"Successful annotations: {statistics_summary['overall_summary']['cases_with_successful_annotation']}")
    print(f"Processing time: {elapsed_time/60:.1f} minutes")
    print(f"Output file: {output_path}")
    
    # Print metrics if available
    if hasattr(annotator, 'get_metrics'):
        metrics = annotator.get_metrics()
        print("\nAPI Metrics:")
        print(f"  Total requests: {metrics.get('total_requests', 0)}")
        print(f"  Success rate: {metrics.get('success_rate', 0):.2%}")
        print(f"  Average duration: {metrics.get('average_duration', 0):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())