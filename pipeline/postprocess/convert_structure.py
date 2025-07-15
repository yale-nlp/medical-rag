import json
import argparse
from pathlib import Path
from tqdm import tqdm

def convert_format(input_path: Path, output_path: Path):
    """
    Converts a JSON result file from the old, verbose format to the new,
    clean, and non-redundant format.

    Args:
        input_path (Path): The path to the source JSON file (old format).
        output_path (Path): The path to the destination JSON file (new format).
    """
    print(f"Loading data from old-format file: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON from {input_path}. Please check the file format.")
        return

    old_reports = source_data.get("all_processed_reports", [])
    if not old_reports:
        print("No cases found in the 'all_processed_reports' list.")
        return

    print(f"Found {len(old_reports)} cases to convert.")

    new_reports_list = []
    for old_case in tqdm(old_reports, desc="Converting cases to new format"):
        # Use .get() for safety in case some keys are missing in older files
        new_case = {
            "case_id": old_case.get("case_id"),
            "original_keywords": old_case.get("original_keywords", []),
            "original_reviewer_report": old_case.get("reviewer_report"), # In old format, this was at the top level
            
            # Rename keys and provide default empty dicts if missing
            "evidence_reranked_papers": old_case.get("reranked_papers_per_keyword", {}),
            "evidence_retrieved_textbook_pages": old_case.get("local_pages_per_keyword", {}),
            "generated_textbook_summaries": old_case.get("generated_textbook_summaries_per_keyword", {}),
            
            "generated_final_feedback": old_case.get("generated_final_ai_feedback", ""),
            "generated_mcqs": old_case.get("generated_mcqs", "")
        }
        new_reports_list.append(new_case)

    # Reconstruct the top-level object with the newly formatted list
    final_output = {
        "pipeline_run_timestamp": source_data.get("pipeline_run_timestamp", "N/A"),
        "pipeline_configuration": source_data.get("pipeline_configuration", {}),
        "all_processed_reports": new_reports_list
    }

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving converted data to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(final_output, f_out, indent=2, ensure_ascii=False)

    print("\nConversion complete.")
    print(f"Successfully converted {len(new_reports_list)} cases.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert old pipeline result JSON files to the new, clean format.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input JSON file (old format)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save the new, converted JSON file."
    )
    
    args = parser.parse_args()

    convert_format(Path(args.input_file), Path(args.output_file))