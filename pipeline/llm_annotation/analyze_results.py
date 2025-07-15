import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import glob
import os

# --- Configuration ---
sns.set_theme(style="whitegrid", context="paper", palette="deep")
plt.rcParams.update({
    'figure.figsize': (12, 7), 'axes.titlesize': 18, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'figure.autolayout': True, 'savefig.dpi': 300
})

MODEL_MAP = {
    "llama3.1-8b": "Llama 3.1-8B", "qwen3-8b": "Qwen3-8B",
    "llama4": "Llama4-Mav-17B", "qwen3-32b": "Qwen3-32B",
    "llama3.3-70b": "Llama 3.3-70B", "medgemma-27b": "MedGemma-27B"
}
JUDGE_MAP = {
    "gemini-2.5pro": "Gemini 2.5 Pro", "gpt-4.1": "GPT-4.1",
    "medgemma": "MedGemma"
}
TABLE_CATEGORIES = {
    "textbook_summary_quality_stats": "Textbook Summary",
    "final_feedback_quality_stats": "Final Feedback",
    "mcq_quality_stats": "MCQ Quality" 
}

# --- Data Loading and Parsing ---

def _to_int_score(value):
    if value is None or value == 0: return np.nan
    try:
        score = int(value)
        return score if 1 <= score <= 5 else np.nan
    except (ValueError, TypeError): return np.nan

def is_human_case_complete(case: dict) -> bool:
    """
    A heuristic to check if a human-annotated case is fully completed.
    It checks if the overall correctness scores (likely the last step) are filled.
    """
    annotations = case.get("annotations", {})
    if annotations.get("skipped", False):
        return True # Skipped cases are considered "handled"
    
    # Check if overall correctness for both experiments are rated (not 0)
    exp1_done = annotations.get("exp1_overall_correctness", 0) != 0
    exp2_done = annotations.get("exp2_overall_correctness", 0) != 0
    
    return exp1_done and exp2_done

def parse_human_case(case, annotator_name):
    records = []
    case_id = case.get("case_id")
    annotations = case.get("annotations", {})
    for kw, score in annotations.get("keyword_appropriateness", {}).items():
        records.append({"case_id": case_id, "annotator": annotator_name, "category": "Keyword Appropriateness", "item": kw, "score": _to_int_score(score)})
    for url, score in annotations.get("paper_relevance", {}).items():
        records.append({"case_id": case_id, "annotator": annotator_name, "category": "Paper Relevance", "item": url, "score": _to_int_score(score)})
    for exp_id, model_name in [("exp1", "Llama 3.3-70B"), ("exp2", "MedGemma-27B")]:
        if s := _to_int_score(annotations.get(f"{exp_id}_summary_quality")):
            records.append({"case_id": case_id, "annotator": annotator_name, "category": "Textbook Summary", "item": f"{model_name}", "score": s})
        if s := _to_int_score(annotations.get(f"{exp_id}_final_feedback_quality")):
            records.append({"case_id": case_id, "annotator": annotator_name, "category": "Final Feedback", "item": f"{model_name}", "score": s})
        for i, score in enumerate(annotations.get(f"{exp_id}_mcq_quality", [])):
            if s := _to_int_score(score):
                records.append({"case_id": case_id, "annotator": annotator_name, "category": "MCQ Quality", "item": f"{model_name}_MCQ_{i+1}", "score": s})
    return records

def parse_llm_case(case, llm_name):
    records = []
    case_id = case.get("case_id")
    annotation = case.get("annotation", {})
    for kw, score in annotation.get("keyword_appropriateness", {}).items():
        records.append({"case_id": case_id, "annotator": llm_name, "category": "Keyword Appropriateness", "item": kw, "score": _to_int_score(score)})
    for url, score in annotation.get("paper_relevance", {}).items():
        records.append({"case_id": case_id, "annotator": llm_name, "category": "Paper Relevance", "item": url, "score": _to_int_score(score)})
    for kw, score in annotation.get("textbook_summary_quality", {}).items():
        records.append({"case_id": case_id, "annotator": llm_name, "category": "Textbook Summary", "item": kw, "score": _to_int_score(score)})
    if s := _to_int_score(annotation.get("final_feedback_quality")):
        records.append({"case_id": case_id, "annotator": llm_name, "category": "Final Feedback", "item": "final_feedback", "score": s})
    for mcq_id, score in annotation.get("mcq_quality", {}).items():
        records.append({"case_id": case_id, "annotator": llm_name, "category": "MCQ Quality", "item": mcq_id, "score": _to_int_score(score)})
    return records

def load_all_data(human_files, llm_paths, overlap_ids_file):
    """Loads all human and LLM annotations, filters by overlap, and returns a tidy DataFrame."""
    with open(overlap_ids_file, 'r') as f:
        overlap_ids = set(str(i) for i in json.load(f))
    print(f"--- Loaded {len(overlap_ids)} overlapping case IDs to filter by. ---")
    
    all_records = []
    
    print("\n--- Loading Human Annotation Files ---")
    for file_path_str in human_files:
        file_path = Path(file_path_str)
        annotator_name = file_path.stem.replace("in_progress_", "")
        overlapping_count = 0
        completed_count = 0
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            for case in data:
                if str(case.get("case_id")) in overlap_ids:
                    overlapping_count += 1
                    if is_human_case_complete(case):
                        completed_count += 1
                    all_records.extend(parse_human_case(case, annotator_name))
            print(f"  [INFO] From '{file_path.name}': Found {overlapping_count} overlapping cases. {completed_count} of them are fully annotated.")
        except Exception as e:
            print(f"  [ERROR] Failed to load or parse {file_path}: {e}")

    print("\n--- Loading LLM Annotation Files ---")
    actual_llm_files = []
    for path_str in llm_paths:
        path = Path(path_str)
        if path.is_dir():
            actual_llm_files.extend(path.glob("**/*_annotated_by_*.json"))
        elif path.is_file():
            actual_llm_files.append(path)

    for file_path in actual_llm_files:
        overlapping_count = 0
        try:
            llm_name = file_path.stem.split("_annotated_by_")[-1].replace("models_", "")
            with open(file_path, 'r') as f:
                data = json.load(f)
            for case in data.get("all_processed_reports", []):
                if str(case.get("case_id")) in overlap_ids:
                    all_records.extend(parse_llm_case(case, llm_name))
                    overlapping_count += 1
            print(f"  [INFO] From '{file_path.name}': Found and loaded {overlapping_count} overlapping cases.")
        except Exception as e:
            print(f"  [ERROR] Failed to load or parse {file_path}: {e}")
                
    df = pd.DataFrame(all_records).dropna(subset=['score'])
    print(f"\n--- Data Loading Summary ---")
    print(f"Total unique overlapping cases with valid scores loaded into DataFrame: {df['case_id'].nunique()}")
    print(f"Total score records loaded: {len(df)}")
    return df

# --- LaTeX Table Generation ---
def generate_common_metrics_table(df, output_path):
    common_cats = ["Keyword Appropriateness", "Paper Relevance"]
    common_df = df[df['category'].isin(common_cats)]
    stats = common_df.groupby(['annotator', 'category'])['score'].agg(['mean', 'std']).unstack()
    latex_string = r"""\begin{table}[t!]
\centering
\begin{tabular}{lcc}
\toprule
\textbf{Annotator} & \textbf{Keyword Appropriateness} & \textbf{Paper Relevance} \\
\midrule
"""
    for annotator in stats.index:
        kw_mean = stats.loc[annotator, ('mean', 'Keyword Appropriateness')]
        kw_std = stats.loc[annotator, ('std', 'Keyword Appropriateness')]
        pr_mean = stats.loc[annotator, ('mean', 'Paper Relevance')]
        pr_std = stats.loc[annotator, ('std', 'Paper Relevance')]
        kw_str = f"{kw_mean:.2f} ($\\pm${kw_std:.2f})" if pd.notna(kw_mean) else "-"
        pr_str = f"{pr_mean:.2f} ($\\pm${pr_std:.2f})" if pd.notna(pr_mean) else "-"
        latex_string += f"\\textbf{{{annotator}}} & {kw_str} & {pr_str} \\\\\n"
    latex_string += r"""\bottomrule
\end{tabular}
\caption{Comparison of common evaluation metrics across human and LLM annotators for overlapping cases. Scores are mean ($\\pm$ std).}
\label{tab:common_metrics}
\end{table*}
"""
    with open(output_path, 'w') as f: f.write(latex_string)
    print(f"\n[SUCCESS] Common metrics LaTeX table saved to: {output_path}")

def generate_human_exp_table(df, output_path):
    human_annotators = [name for name in df['annotator'].unique() if 'Gupta' in name or 'jtc' in name]
    human_df = df[df['annotator'].isin(human_annotators)]
    exp_cats = ["Textbook Summary", "MCQ Quality", "Final Feedback"]
    exp_df = human_df[human_df['category'].isin(exp_cats)].copy()
    
    exp_df['Experiment'] = exp_df['item'].apply(lambda x: "Llama 3.3-70B" if "Llama" in str(x) else "MedGemma-27B")
    
    stats = exp_df.groupby(['annotator', 'Experiment', 'category'])['score'].agg(['mean', 'std']).unstack()
    
    latex_string = r"""\begin{table*}[t!]
\centering
\begin{tabular}{llcccc}
\toprule
\textbf{Annotator} & \textbf{Experiment} & \textbf{Textbook Summary} & \textbf{MCQ Quality} & \textbf{Final Feedback} & \textbf{Average} \\
\midrule
"""
    for annotator in human_annotators:
        for exp in ["Llama 3.3-70B", "MedGemma-27B"]:
            try:
                row = stats.loc[(annotator, exp)]
                ts_mean, ts_std = row.get(('mean', 'Textbook Summary'), np.nan), row.get(('std', 'Textbook Summary'), np.nan)
                mcq_mean, mcq_std = row.get(('mean', 'MCQ Quality'), np.nan), row.get(('std', 'MCQ Quality'), np.nan)
                ff_mean, ff_std = row.get(('mean', 'Final Feedback'), np.nan), row.get(('std', 'Final Feedback'), np.nan)
                avg_score = np.nanmean([ts_mean, mcq_mean, ff_mean])
                ts_str = f"{ts_mean:.2f} ($\\pm${ts_std:.2f})" if pd.notna(ts_mean) else "-"
                mcq_str = f"{mcq_mean:.2f} ($\\pm${mcq_std:.2f})" if pd.notna(mcq_mean) else "-"
                ff_str = f"{ff_mean:.2f} ($\\pm${ff_std:.2f})" if pd.notna(ff_mean) else "-"
                avg_str = f"\\textbf{{{avg_score:.2f}}}" if pd.notna(avg_score) else "-"
                latex_string += f"\\textbf{{{annotator}}} & {exp} & {ts_str} & {mcq_str} & {ff_str} & {avg_str} \\\\\n"
            except KeyError: continue
        latex_string += r"\midrule" + "\n"
    latex_string = latex_string.rsplit(r'\midrule', 1)[0]
    latex_string += r"""\bottomrule
\end{tabular}
\caption{Human evaluation scores for experiment-specific components (Llama 3.3-70B vs. MedGemma-27B). Scores are mean ($\\pm$ std).}
\label{tab:human_exp_eval}
\end{table*}
"""
    with open(output_path, 'w') as f: f.write(latex_string)
    print(f"[SUCCESS] Human experiment LaTeX table saved to: {output_path}")

# --- Plotting Functions ---
def plot_score_distributions(df, output_dir):
    print("\n--- Generating Score Distribution Plots ---")
    g = sns.catplot(data=df, x='category', y='score', hue='annotator', kind='violin', col='category', col_wrap=3, sharex=False, sharey=True, height=5, aspect=1.2, palette='plasma', inner='quartile')
    g.fig.suptitle('Score Distribution by Annotator and Category', y=1.03, fontsize=20)
    g.set_axis_labels("", "Score (1-5)")
    g.set_titles("{col_name}")
    g.set_xticklabels(rotation=30, ha='right')
    g.set(ylim=(0.5, 5.5))
    for ext in ['png', 'pdf']: plt.savefig(output_dir / f"plot_score_distributions.{ext}")
    print(f"[SUCCESS] Score distribution plots saved.")

def plot_human_agreement_heatmap(df, output_dir):
    print("\n--- Generating Human Agreement Heatmap ---")
    human_annotators = [name for name in df['annotator'].unique() if 'Gupta' in name or 'jtc' in name]
    if len(human_annotators) != 2:
        print("[WARN] Exactly two human annotators required for agreement heatmap. Skipping.")
        return
    h1_name, h2_name = human_annotators[0], human_annotators[1]
    pivot_df = df.pivot_table(index=['category', 'item'], columns='annotator', values='score').dropna()
    for category in pivot_df.index.get_level_values('category').unique():
        cat_df = pivot_df.loc[category]
        if len(cat_df) < 2: continue
        confusion_matrix = pd.crosstab(cat_df[h1_name], cat_df[h2_name])
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu', cbar=True)
        plt.title(f'Inter-Annotator Agreement: {category}')
        plt.xlabel(f'Annotator: {h2_name}')
        plt.ylabel(f'Annotator: {h1_name}')
        for ext in ['png', 'pdf']: plt.savefig(output_dir / f"plot_agreement_heatmap_{category.replace(' ', '_')}.{ext}")
    print(f"[SUCCESS] Human agreement heatmaps saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze human and LLM annotations.")
    parser.add_argument("--human_files", nargs='+', required=True, help="Paths to human annotation files (e.g., in_progress_Gupta.json).")
    parser.add_argument("--llm_files", nargs='+', required=True, help="Paths to LLM annotation files or directories containing them. Can use wildcards.")
    parser.add_argument("--overlap_ids", required=True, help="Path to the JSON file with overlapping case IDs.")
    parser.add_argument("--output_dir", required=True, help="Directory to save tables and plots.")
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    master_df = load_all_data(args.human_files, args.llm_files, args.overlap_ids)
    
    if master_df.empty:
        print("\n[FATAL] No overlapping data found to analyze. Exiting.")
        exit(1)

    generate_common_metrics_table(master_df, output_path / "table_common_metrics.tex")
    generate_human_exp_table(master_df, output_path / "table_human_experiments.tex")
    plot_score_distributions(master_df, output_path)
    plot_human_agreement_heatmap(master_df, output_path)

    print("\n--- All analysis tasks complete! ---")