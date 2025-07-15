import json
import re
import argparse
from pathlib import Path
import numpy as np
from collections import Counter

def _to_int_score(value) -> int | None:
    if value is None: return None
    try:
        score = int(value)
        return score if 1 <= score <= 5 else None
    except (ValueError, TypeError): return None

def calculate_statistics(all_annotated_cases: list) -> dict:
    print("\n--- Calculating Final Statistics for OpenAI Results ---")
    stats = {"overall_summary": {"total_cases_processed": len(all_annotated_cases), "cases_with_successful_annotation": 0}}
    scores = {"keyword_appropriateness": [], "paper_relevance": [], "textbook_summary_quality": [], "mcq_quality": [], "final_feedback_quality": [], "holistic_quality": []}
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
        if (s := _to_int_score(annotation.get("holistic_quality"))) is not None: scores["holistic_quality"].append(s)
    for category, score_list in scores.items():
        if not score_list:
            stats[category + "_stats"] = "No valid scores available."
            continue
        stats[category + "_stats"] = {"count": len(score_list), "mean": float(np.mean(score_list)), "std_dev": float(np.std(score_list)), "min": int(np.min(score_list)), "max": int(np.max(score_list)), "distribution": dict(sorted(Counter(score_list).items()))}
    return stats

def main():
    parser = argparse.ArgumentParser(description="Process OpenAI Batch API results and merge them.")
    parser.add_argument("--batch_output_file", required=True, help="Path to the downloaded .jsonl output file from OpenAI.")
    parser.add_argument("--intermediate_file", required=True, help="Path to the JSON file created by run_evaluations.py for the OpenAI batch job.")
    parser.add_argument("--final_output_file", required=True, help="Path for the final, fully annotated output JSON file.")
    args = parser.parse_args()

    print("--- Merging OpenAI Batch Results ---")
    with open(args.intermediate_file, 'r', encoding='utf-8') as f:
        final_data = json.load(f).get("original_data")
    
    case_map = {case['case_id']: case for case in final_data['all_processed_reports']}
    
    with open(args.batch_output_file, 'r', encoding='utf-8') as f:
        for line in f:
            batch_line = json.loads(line)
            custom_id = batch_line.get("custom_id")
            response_body = batch_line.get("response", {}).get("body", {})
            if not custom_id or not response_body: continue

            case_id, category, item_key = custom_id.split("__")
            if case_id not in case_map: continue
            
            case = case_map[case_id]
            if "annotation" not in case: case["annotation"] = {}

            content = response_body.get("choices", [{}])[0].get("message", {}).get("content", "")
            score_match = re.search(r'\d', content)
            score = int(score_match.group(0)) if score_match else None

            if category == "keyword": case["annotation"].setdefault("keyword_appropriateness", {})[item_key] = score
            elif category == "paper": case["annotation"].setdefault("paper_relevance", {})[item_key] = score
            elif category == "summary": case["annotation"].setdefault("textbook_summary_quality", {})[item_key] = score
            elif category == "mcq": case["annotation"].setdefault("mcq_quality", {})[item_key] = score
            elif category == "final_feedback": case["annotation"]["final_feedback_quality"] = score
            elif category == "holistic": case["annotation"]["holistic_quality"] = score

    final_data["statistics_summary"] = calculate_statistics(final_data['all_processed_reports'])

    with open(args.final_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n[SUCCESS] Final annotated file with OpenAI results and statistics saved to: {args.final_output_file}")

if __name__ == "__main__":
    main()