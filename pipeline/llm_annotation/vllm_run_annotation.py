import json
import re
import os
import argparse
import asyncio
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from vllm_annotator import get_annotator, GeminiAnnotator, OpenAIAnnotator
import numpy as np
from collections import Counter

def _to_int_score(value) -> int | None:
    if value is None: return None
    try:
        score = int(value)
        return score if 1 <= score <= 5 else None
    except (ValueError, TypeError): return None

def calculate_and_append_statistics(all_annotated_cases: list) -> dict:
    print("\n--- Calculating Final Statistics ---")
    stats = {"overall_summary": {"total_cases_processed": len(all_annotated_cases), "cases_with_successful_annotation": 0}}
    scores = {"keyword_appropriateness": [], "paper_relevance": [], "textbook_summary_quality": [], "mcq_quality": [], "final_feedback_quality": [], "overall_quality": []}
    correlation_data = {"rerank_scores": [], "paper_relevance_scores": []}
    for case in all_annotated_cases:
        annotation = case.get("annotation")
        if not annotation or "error" in annotation: continue
        stats["overall_summary"]["cases_with_successful_annotation"] += 1
        for score in annotation.get("keyword_appropriateness", {}).values():
            if (s := _to_int_score(score)) is not None: scores["keyword_appropriateness"].append(s)
        for score in annotation.get("paper_relevance", {}).values():
            if (s := _to_int_score(score)) is not None: scores["paper_relevance"].append(s)
        for score in annotation.get("textbook_summary_quality", {}).values():
            if (s := _to_int_score(score)) is not None: scores["textbook_summary_quality"].append(s)
        for score in annotation.get("mcq_quality", {}).values():
            if (s := _to_int_score(score)) is not None: scores["mcq_quality"].append(s)
        if (s := _to_int_score(annotation.get("final_feedback_quality"))) is not None: scores["final_feedback_quality"].append(s)
        if (s := _to_int_score(annotation.get("overall_quality"))) is not None: scores["overall_quality"].append(s)
        for kw_papers in case.get("evidence_reranked_papers", {}).values():
            for paper in kw_papers:
                relevance_score = _to_int_score(annotation.get("paper_relevance", {}).get(paper.get("url")))
                if relevance_score is not None and paper.get("rerank_score") is not None:
                    correlation_data["rerank_scores"].append(paper.get("rerank_score"))
                    correlation_data["paper_relevance_scores"].append(relevance_score)
    for category, score_list in scores.items():
        if not score_list:
            stats[category + "_stats"] = "No valid scores available."
            continue
        stats[category + "_stats"] = {"count": len(score_list), "mean": float(np.mean(score_list)), "std_dev": float(np.std(score_list)), "min": int(np.min(score_list)), "max": int(np.max(score_list)), "distribution": dict(sorted(Counter(score_list).items()))}
    if len(correlation_data["rerank_scores"]) > 1:
        correlation_matrix = np.corrcoef(correlation_data["rerank_scores"], correlation_data["paper_relevance_scores"])
        stats.setdefault("paper_relevance_stats", {})["correlation_with_rerank_score"] = {"pearson_coefficient": float(correlation_matrix[0, 1]), "data_points_count": len(correlation_data["rerank_scores"])}
    return stats

def run_sync_annotation_on_case(annotator, case_data: dict) -> dict:
    case_id, original_report, original_keywords = case_data.get("case_id"), case_data.get("original_reviewer_report"), case_data.get("original_keywords", [])
    print(f"\n--- Annotating Case ID: {case_id} (Sync) ---")
    if not all([original_report, original_keywords]): return {"error": "Missing data"}
    results = {"keyword_appropriateness": {}, "paper_relevance": {}, "textbook_summary_quality": {}, "mcq_quality": {}, "final_feedback_quality": None, "overall_quality": None}
    for kw in original_keywords:
        kw_ctx = {"original_report": original_report, "keyword": kw}
        results["keyword_appropriateness"][kw] = _to_int_score(annotator.evaluate(annotator._construct_prompt("keyword_appropriateness_evaluation", kw_ctx)).get("score"))
        for paper in case_data.get("evidence_reranked_papers", {}).get(kw, []):
            paper_ctx = {**kw_ctx, "current_paper_title": paper.get("title", "N/A"), "current_paper_abstract": paper.get("abstract", "N/A")}
            results["paper_relevance"][paper.get("url")] = _to_int_score(annotator.evaluate(annotator._construct_prompt("reranked_results_evaluation", paper_ctx)).get("score"))
        summary = case_data.get("generated_textbook_summaries", {}).get(kw)
        if summary: results["textbook_summary_quality"][kw] = _to_int_score(annotator.evaluate(annotator._construct_prompt("textbook_summary_evaluation", {**kw_ctx, "textbook_summary_text": summary})).get("score"))
    mcqs_text = case_data.get("generated_mcqs", "")
    if mcqs_text:
        for i, m in enumerate(re.finditer(r"Q\d+\.\s*(?P<q>[\s\S]+?)\nAnswer:\s*(?P<a>[^\n]+)\nExplanation:\s*(?P<e>[\s\S]+?)(?=\nQ\d+|$)", mcqs_text, re.I)):
            mcq_ctx = {"keyword": "overall case", "mcq_question_text": m.group('q').strip(), "mcq_answer_text": m.group('a').strip(), "mcq_explanation_text": m.group('e').strip()}
            results["mcq_quality"][f"MCQ_{i+1}"] = _to_int_score(annotator.evaluate(annotator._construct_prompt("mcq_evaluation", mcq_ctx)).get("score"))
    feedback_text = case_data.get("generated_final_feedback", "")
    if feedback_text:
        fb_ctx = {"original_report": original_report, "keywords_list_str": "\n- ".join(original_keywords), "full_generated_feedback": feedback_text}
        results["final_feedback_quality"] = _to_int_score(annotator.evaluate(annotator._construct_prompt("final_feedback_synthesis_evaluation", fb_ctx)).get("score"))
    all_papers = "\n\n".join([f"Title: {p.get('title')}\nAbs: {p.get('abstract')}" for kw_ps in case_data.get("evidence_reranked_papers", {}).values() for p in kw_ps])
    overall_ctx = {"original_report": original_report, "keywords_list_str": "\n- ".join(original_keywords), "all_reranked_papers_text": all_papers or "N/A", "all_mcqs_text": mcqs_text or "N/A", "full_generated_feedback": feedback_text or "N/A"}
    results["overall_quality"] = _to_int_score(annotator.evaluate(annotator._construct_prompt("overall_case_evaluation", overall_ctx)).get("score"))
    return results

def generate_prompts_for_case(case_data: dict, annotator_instance) -> list:
    case_id, original_report, original_keywords = case_data.get("case_id"), case_data.get("original_reviewer_report"), case_data.get("original_keywords", [])
    if not all([case_id, original_report, original_keywords]): return []
    prompts = []
    paper_fallback_counter = 0
    for kw in original_keywords:
        kw_context = {"original_report": original_report, "keyword": kw}
        prompts.append((f"{case_id}__keyword_appropriateness__{kw}", annotator_instance._construct_prompt("keyword_appropriateness_evaluation", kw_context)))
        for paper in case_data.get("evidence_reranked_papers", {}).get(kw, []):
            paper_id = paper.get("url") or f"no_url_paper_{paper_fallback_counter}"
            paper_fallback_counter += 1 if not paper.get("url") else 0
            paper_context = {**kw_context, "current_paper_title": paper.get("title", "N/A"), "current_paper_abstract": paper.get("abstract", "N/A")}
            prompts.append((f"{case_id}__paper_relevance__{paper_id}", annotator_instance._construct_prompt("reranked_results_evaluation", paper_context)))
        summary = case_data.get("generated_textbook_summaries", {}).get(kw)
        if summary: prompts.append((f"{case_id}__textbook_summary_quality__{kw}", annotator_instance._construct_prompt("textbook_summary_evaluation", {**kw_context, "textbook_summary_text": summary})))
    mcqs_text = case_data.get("generated_mcqs", "")
    if mcqs_text:
        for i, match in enumerate(re.finditer(r"Q\d+\.\s*(?P<q>[\s\S]+?)\nAnswer:\s*(?P<a>[^\n]+)\nExplanation:\s*(?P<e>[\s\S]+?)(?=\nQ\d+|$)", mcqs_text, re.I)):
            mcq_context = {"keyword": "overall case", "mcq_question_text": match.group('q').strip(), "mcq_answer_text": match.group('a').strip(), "mcq_explanation_text": match.group('e').strip()}
            prompts.append((f"{case_id}__mcq_quality__MCQ_{i+1}", annotator_instance._construct_prompt("mcq_evaluation", mcq_context)))
    feedback_text = case_data.get("generated_final_feedback", "")
    if feedback_text:
        feedback_context = {"original_report": original_report, "keywords_list_str": "\n- ".join(original_keywords), "full_generated_feedback": feedback_text}
        prompts.append((f"{case_id}__final_feedback_quality__-", annotator_instance._construct_prompt("final_feedback_synthesis_evaluation", feedback_context)))
    all_papers_text = "\n\n".join([f"Title: {p.get('title', 'N/A')}\nAbstract: {p.get('abstract', 'N/A')}" for kw_papers in case_data.get("evidence_reranked_papers", {}).values() for p in kw_papers])
    overall_context = {"original_report": original_report, "keywords_list_str": "\n- ".join(original_keywords), "all_reranked_papers_text": all_papers_text or "N/A", "all_mcqs_text": mcqs_text or "N/A", "full_generated_feedback": feedback_text or "N/A"}
    prompts.append((f"{case_id}__overall_quality__-", annotator_instance._construct_prompt("overall_case_evaluation", overall_context)))
    return prompts

async def run_async_gemini_evaluation(annotator: GeminiAnnotator, all_cases: list, concurrency_limit: int = 100):
    print(f"--- Gemini Annotator: Preparing all prompts for async execution (Concurrency: {concurrency_limit}) ---")
    
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def controlled_evaluate(prompt_text):
        async with semaphore:
            return await annotator.evaluate_async(prompt_text)

    all_tasks, all_contexts = [], []
    for case in tqdm(all_cases, desc="Generating Prompts"):
        prompts_for_case = generate_prompts_for_case(case, annotator)
        for custom_id, prompt_text in prompts_for_case:
            all_tasks.append(controlled_evaluate(prompt_text))
            all_contexts.append(custom_id)
    
    print(f"\n--- Sending {len(all_tasks)} requests to Gemini API with controlled concurrency ---")
    all_results = await async_tqdm.gather(*all_tasks)
    
    print("\n--- Mapping results back to cases ---")
    case_map = {case['case_id']: case for case in all_cases}
    for context_id, result in tqdm(zip(all_contexts, all_results), total=len(all_contexts), desc="Processing results"):
        case_id, category, item_key = context_id.split("__")
        if case_id not in case_map: continue
        case = case_map[case_id]
        if "annotation" not in case: case["annotation"] = {}
        score = _to_int_score(result.get("score"))
        if category in ["final_feedback_quality", "overall_quality"]:
            case["annotation"][category] = score
        else:
            case["annotation"].setdefault(category, {})[item_key] = score
    return all_cases

async def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge evaluations.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="annotation_configs.json")
    parser.add_argument("--limit", type=int, default=None)
    # [ADDED] Argument for concurrency limit
    parser.add_argument("--concurrency", type=int, default=100, help="Concurrency limit for Gemini API calls.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.config_file, "r", encoding="utf-8") as f: config = json.load(f)
    
    annotator_choice = config.get("annotator_choice", "openai").lower()
    if annotator_choice == "huggingface":
        os.environ["CUDA_VISIBLE_DEVICES"] = config.get("cuda_visible_devices", "")
    
    annotator = get_annotator(args.config_file)
    model_name = config.get(f"{annotator_choice}_annotator_settings", {}).get("model_name", config.get(f"{annotator_choice}_annotator_settings", {}).get("model"))
    
    with open(args.input_file, "r", encoding="utf-8") as f: input_data = json.load(f)
    all_cases = input_data.get("all_processed_reports", [])
    if args.limit: all_cases = all_cases[:args.limit]

    if isinstance(annotator, GeminiAnnotator):
        all_cases = await run_async_gemini_evaluation(annotator, all_cases, concurrency_limit=args.concurrency)
    elif isinstance(annotator, OpenAIAnnotator):
        print("--- OpenAI Annotator: Submitting Batch Job ---")
        all_prompts = []
        for case in tqdm(all_cases, desc="Generating Prompts for OpenAI Batch"):
            all_prompts.extend(generate_prompts_for_case(case, annotator))
        batch_id = annotator.prepare_and_submit_batch(all_prompts, output_dir)
        print(f"\n{'='*60}\nIMPORTANT: OpenAI BATCH JOB SUBMITTED\nBatch ID: {batch_id}\n{'='*60}\n")
        intermediate_data = {"batch_id": batch_id, "original_data": input_data}
        output_path = output_dir / f"{Path(args.input_file).stem}_openai_batch_pending_{batch_id}.json"
        with open(output_path, "w", encoding="utf-8") as f: json.dump(intermediate_data, f, indent=2)
        print(f"Batch job info and original data saved to: {output_path}")
        return
    else: # HuggingFace
        print(f"--- HuggingFace Annotator: Running Sync Evaluations ---")
        for case in tqdm(all_cases, desc=f"Annotating with {model_name} (Sync)"):
            case['annotation'] = run_sync_annotation_on_case(annotator, case)

    statistics_summary = calculate_and_append_statistics(all_cases)
    output_filename = f"{Path(args.input_file).stem}_annotated_by_{re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name)}.json"
    final_output_data = {"statistics_summary": statistics_summary, "pipeline_configuration": input_data.get("pipeline_configuration", {}), "all_processed_reports": all_cases}
    with open(output_dir / output_filename, "w", encoding="utf-8") as f: json.dump(final_output_data, f, indent=2, ensure_ascii=False)
    print(f"\n[SUCCESS] Annotation complete. Final results saved to: {output_dir / output_filename}")

if __name__ == "__main__":
    asyncio.run(main())