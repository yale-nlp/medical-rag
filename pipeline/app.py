import gradio as gr
import json
import subprocess
import time
from pathlib import Path
import os
import html
from typing import Dict, Any, List
import logging
import re
import argparse
from dotenv import load_dotenv

# --- Constants and Configuration ---
CONFIG_PATH = Path("configs/gradio_config.json")
TEMP_CONFIG_PATH = Path("configs/temp_gradio_run.json")
ALL_CASES = []
DEMO_RESULT_FILE_PATH = Path("./result/gradio_test/demo_results.json")

load_dotenv()

# --- Custom CSS ---
CUSTOM_CSS = """
.gradio-container { font-family: 'IBM Plex Sans', sans-serif; }
.prose { font-size: 16px !important; line-height: 1.6 !important; }
.section-header > .label-wrap {
    background: linear-gradient(to right, #f08c00, #e67700) !important; border: 1px solid #d96e00 !important;
    font-size: 1rem !important; font-weight: 700 !important; color: white !important; padding: 12px 16px !important;
    text-align: left !important; border-radius: 8px !important; margin-bottom: 8px !important;
    text-transform: uppercase; letter-spacing: 0.8px; transition: all 0.2s ease-in-out;
}
.section-header > .label-wrap:hover { background: linear-gradient(to right, #e67700, #d96e00) !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.mcq-block { margin-bottom: 20px; border-top: 2px solid #f1f3f5; padding-top: 15px; }
.final-feedback-box { padding: 15px; background-color: #f8f9fa; border-radius: 8px; }
"""

# --- Helper Functions ---

def load_cases_from_file(keyword_file_path: str) -> List[Dict]:
    global ALL_CASES
    try:
        path = Path(keyword_file_path)
        if not path.exists():
            logging.warning(f"Keyword file not found: '{keyword_file_path}'")
            ALL_CASES = []
        else:
            with open(path, 'r', encoding='utf-8') as f:
                ALL_CASES = [json.loads(line) for line in f if line.strip()]
            logging.info(f"Loaded {len(ALL_CASES)} cases from {keyword_file_path}")
    except Exception as e:
        logging.error(f"Error loading cases: {e}")
        ALL_CASES = []
    return [c.get("case_id") for c in ALL_CASES]

def update_case_dropdown():
    case_ids = [c.get("case_id") for c in ALL_CASES]
    return gr.Dropdown(choices=case_ids, value=case_ids[0] if case_ids else None)

def setup_and_load_initial_config():
    config_path = CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        logging.warning(f"Config file not found. Creating a default config at: {config_path}")
        default_config = {
            "keyword_file": "/home/dj475/yalenlp/medical-rag/data/input_files/mimic-iv-note_sampled2000.jsonl",
            "feedback_dir": "results/gradio_run",
            "hf_embedding_model": "Qwen/Qwen3-Embedding-8B",
            "embedding_model_device": "cpu",
            "retrievers": ["local", "semantic", "pubmed"],
            "local_index_path": "/home/dj475/yalenlp/medical-rag/old_pipeline/data/embedded_pages_hf_gpu_Qwen_Qwen3-Embedding-8B.json",
            "s2_retrieved_file": None, "pubmed_retrieved_file": None,
            "service_map": { "reranker": "reranker_service", "textbook_summary": "generation_service", "final_report": "generation_service", "mcq_generation": "generation_service" },
            "services": {
                "reranker_service": { "model": "Qwen/Qwen3-Reranker-8B", "gpu_id": "0", "tensor_parallel_size": 1, "max_model_len": 512, "rerank_candidates_num": 2, "qwen_reranker_params": {"qwen_instruction_task": "Find the most relevant document for the medical query."} },
                "generation_service": { "model": "google/medgemma-27b-text-it", "gpu_id": "1,2", "tensor_parallel_size": 2, "quantization": None, "max_model_len": 8192, "generation_params": { "textbook_summary": { "max_tokens": 512, "temperature": 0.1, "top_p": 0.9, "stop_tokens": ["<|eot_id|>"] }, "final_report": { "max_tokens": 2048, "temperature": 0.2, "top_p": 0.9, "stop_tokens": ["<|eot_id|>"] }, "mcq_generation": { "max_tokens": 1024, "temperature": 0.2, "top_p": 0.9, "stop_tokens": ["<|eot_id|>"] } } }
            },
            "system": { "textbook_summary": { "system_persona": "You are a concise and accurate radiology assistant...", "user_instruction_template": "Please summarize..." }, "final_feedback": { "system_persona": "You are an expert radiology AI assistant...", "user_instruction_template": "### Primary Diagnostic Keywords\n- {keywords_list_str}\n\n..." }, "questioner": { "system_persona": "You are a specialized AI assistant for creating MCQs...", "user_instruction_template": "### Primary Diagnostic Keywords to Focus On:\n- {keywords_list_str}\n\n..." } }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        return default_config
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_retrieval_md(reranked_papers, local_pages):
    md = ""
    all_keywords = set(reranked_papers.keys()) | set(local_pages.keys())
    if not all_keywords: return "<i>No retrieval results available.</i>"
    for kw in sorted(list(all_keywords)):
        md += f"#### Keyword: `{kw}`\n"
        md += "##### **Web Evidence (Reranked)**\n"
        papers = reranked_papers.get(kw, [])
        if papers:
            for p in papers:
                title = html.escape(p.get('title') or 'N/A')
                abstract = html.escape(p.get('abstract') or 'N/A')
                md += f"""- **Title:** {title}\n  - **Abstract:** <small>{abstract}</small>\n  - **Source:** {p.get('source', 'N/A')} | **Score:** <span style='color: #d9534f;'>{p.get('rerank_score', 0.0):.4f}</span> | [Link]({p.get('url', '#')})\n"""
        else: md += "_No web documents found or reranked._\n"
        md += "\n##### **Local Evidence (Textbook)**\n"
        pages = local_pages.get(kw, [])
        if pages:
            for p in pages:
                text_preview = html.escape((p.get('text') or '')[:200] + '...')
                md += f"- **Page Index:** {p.get('index', 'N/A')}\n  - **Content:** <small>{text_preview}</small>\n"
        else: md += "_No local documents found._\n"
    return md

def format_summary_md(summaries):
    if not summaries: return "<i>No summaries generated.</i>"
    md = ""
    for kw, summary in summaries.items():
        md += f"#### Keyword: `{kw}`\n> {html.escape(summary or 'N/A')}\n"
    return md

def format_final_feedback_md(final_report):
    if not final_report: return "<i>No final feedback generated.</i>"
    return final_report

def format_mcqs_md(mcqs):
    if not mcqs: return "<i>No MCQs generated.</i>"
    mcq_html = ""
    mcqs_text = mcqs or ""
    question_blocks = re.finditer(r"(Q\d+\..*?)(?=Q\d+\.|\Z)", mcqs_text, re.DOTALL)
    found_mcqs = False
    for match in question_blocks:
        found_mcqs = True
        full_question_text = match.group(1).strip()
        lines = full_question_text.split('\n')
        q_text, options, answer, explanation = "", [], "", ""
        for line in lines:
            line = line.strip()
            if not line: continue
            if re.match(r'Q\d+\.', line): q_text = line
            elif re.match(r'^[A-D]\.', line): options.append(line)
            elif line.lower().startswith('answer:'): answer = line
            elif line.lower().startswith('explanation:'): explanation = line
        mcq_html += f"<div class='mcq-block'>"
        mcq_html += f"<p><strong>{html.escape(q_text)}</strong></p>"
        for opt in options: mcq_html += f"<p style='margin-left: 20px;'>{html.escape(opt)}</p>"
        answer_text = answer.split(':', 1)[-1].strip() if ':' in answer else answer
        explanation_text = explanation.split(':', 1)[-1].strip() if ':' in explanation else explanation
        mcq_html += f"<p><strong>Answer:</strong> <span style='color: #28a745; font-weight: bold;'>{html.escape(answer_text)}</span></p>"
        mcq_html += f"<blockquote>{html.escape(explanation_text)}</blockquote></div>"
    if not found_mcqs:
        mcq_html += f"<div>{html.escape(mcqs_text).replace(chr(10), '<br>')}</div>"
    return mcq_html

def parse_and_format_output(config: Dict[str, Any], case_id: str):
    if not case_id: return "Please select a case first.", "", "", "", "", "Error: No case selected."
    output_path = None
    if DEMO_MODE_ENABLED:
        output_path = DEMO_RESULT_FILE_PATH
        if not output_path.exists():
            error_msg = f"DEMO FILE NOT FOUND: Please place your pre-generated results file at '{output_path}'."
            return error_msg, "", "", "", "", error_msg
    else:
        feedback_dir = Path(config.get("feedback_dir", "results/gradio_run"))
        keyword_file_stem = Path(config["keyword_file"]).stem
        output_filename = f"{keyword_file_stem}_local_vllm_feedback.json"
        output_path = feedback_dir / output_filename
        if not output_path.exists():
            error_msg = f"Output file not found at {output_path}. Please run the pipeline in normal mode first."
            return error_msg, "", "", "", "", error_msg
    try:
        with open(output_path, 'r', encoding='utf-8') as f: results = json.load(f)
        case_result = next((r for r in results.get("all_processed_reports", []) if r.get("case_id") == case_id), None)
        if not case_result:
            error_msg = f"No results found for Case ID: {case_id} in the result file '{output_path.name}'."
            return error_msg, "", "", "", "", error_msg
        report_text = (case_result.get('original_reviewer_report') or 'N/A').replace('\n', '\n> ')
        input_md = f"#### Original Reviewer Report\n> {html.escape(report_text)}\n\n#### Primary Keywords\n- `{'`, `'.join(case_result.get('original_keywords', []))}`"
        retrieval_md = format_retrieval_md(case_result.get("evidence_reranked_papers", {}), case_result.get("evidence_retrieved_textbook_pages", {}))
        summary_md = format_summary_md(case_result.get("generated_textbook_summaries", {}))
        final_feedback_md = format_final_feedback_md(case_result.get('generated_final_feedback'))
        mcqs_md = format_mcqs_md(case_result.get('generated_mcqs'))
        return input_md, retrieval_md, summary_md, final_feedback_md, mcqs_md, "OK"
    except Exception as e:
        logging.error(f"Error parsing output file: {e}", exc_info=True)
        error_msg = f"An error occurred while parsing output: {e}"
        return error_msg, "", "", "", "", error_msg

def _assemble_config_from_ui(*args: Any) -> Dict[str, Any]:
    (
        keyword_file, feedback_dir, hf_embedding_model, retrievers, local_index_path,
        s2_retrieved_file, pubmed_retrieved_file,
        reranker_model, reranker_gpu, reranker_tp, rerank_k, reranker_instr,
        gen_model, gen_gpu, gen_tp,
        ts_max_tokens, ts_temp, ts_top_p,
        ff_max_tokens, ff_temp, ff_top_p,
        mcq_max_tokens, mcq_temp, mcq_top_p,
        ts_persona, ts_template, ff_persona, ff_template, q_persona, q_template
    ) = args
    config = {
        "keyword_file": keyword_file, "feedback_dir": feedback_dir,
        "hf_embedding_model": hf_embedding_model, "embedding_model_device": "cpu",
        "retrievers": retrievers, "local_index_path": local_index_path,
        "s2_retrieved_file": s2_retrieved_file if s2_retrieved_file else None,
        "pubmed_retrieved_file": pubmed_retrieved_file if pubmed_retrieved_file else None,
        "service_map": { "reranker": "reranker_service", "textbook_summary": "generation_service", "final_report": "generation_service", "mcq_generation": "generation_service" },
        "services": {
            "reranker_service": { "model": reranker_model, "gpu_id": reranker_gpu, "tensor_parallel_size": reranker_tp, "max_model_len": 512, "rerank_candidates_num": rerank_k, "qwen_reranker_params": {"qwen_instruction_task": reranker_instr} },
            "generation_service": { "model": gen_model, "gpu_id": gen_gpu, "tensor_parallel_size": gen_tp, "quantization": None, "max_model_len": 8192, "generation_params": { "textbook_summary": {"max_tokens": ts_max_tokens, "temperature": ts_temp, "top_p": ts_top_p, "stop_tokens": ["<|eot_id|>"]}, "final_report": {"max_tokens": ff_max_tokens, "temperature": ff_temp, "top_p": ff_top_p, "stop_tokens": ["<|eot_id|>"]}, "mcq_generation": {"max_tokens": mcq_max_tokens, "temperature": mcq_temp, "top_p": mcq_top_p, "stop_tokens": ["<|eot_id|>"]} } }
        },
        "system": { "textbook_summary": {"system_persona": ts_persona, "user_instruction_template": ts_template}, "final_feedback": {"system_persona": ff_persona, "user_instruction_template": ff_template}, "questioner": {"system_persona": q_persona, "user_instruction_template": q_template} }
    }
    return config

def simulate_pipeline_run(case_id, progress=gr.Progress(track_tqdm=True), *all_config_args):
    yield (
        gr.Button(interactive=False, value="Simulating..."),
        gr.Button(interactive=False, value="Simulating..."),
        gr.Textbox(value=f"Starting demo simulation for case '{case_id}'...", interactive=False),
        "", "", "", "", "",
        f"--- DEMO MODE ---\nStarting simulation for case '{case_id}'.\n"
    )
    runtime_config = _assemble_config_from_ui(*all_config_args)
    input_md, retrieval_md, summary_md, final_feedback_md, mcqs_md, status = parse_and_format_output(runtime_config, case_id)
    if status != "OK":
        gr.Error(status)
        yield (
            gr.Button(interactive=True, value="🚀 Run Single Case"),
            gr.Button(interactive=True, value="Run All Cases"),
            gr.Textbox(value="Demo Error", interactive=False),
            gr.Markdown(), gr.Markdown(), gr.Markdown(), gr.Markdown(), gr.Markdown(),
            status
        )
        return
    log_content = f"--- DEMO MODE ---\nStarting simulation for case '{case_id}'.\n"
    progress(0.20, desc="Step 1/5: Displaying Input...")
    log_content += "Step 1/5: Input and keywords loaded.\n"
    yield (gr.update(), gr.update(), "Step 1/5: Displaying Input...", input_md, gr.update(), gr.update(), gr.update(), gr.update(), log_content)
    time.sleep(4)
    progress(0.40, desc="Step 2/5: Simulating Retrieval & Reranking...")
    log_content += "Step 2/5: Retrieval and Reranking simulation complete.\n"
    yield (gr.update(), gr.update(), "Step 2/5: Simulating Retrieval & Reranking...", gr.update(), retrieval_md, gr.update(), gr.update(), gr.update(), log_content)
    time.sleep(4)
    progress(0.60, desc="Step 3/5: Simulating Summary Generation...")
    log_content += "Step 3/5: Intermediate Summary generation simulation complete.\n"
    yield (gr.update(), gr.update(), "Step 3/5: Simulating Summary Generation...", gr.update(), gr.update(), summary_md, gr.update(), gr.update(), log_content)
    time.sleep(4)
    progress(0.80, desc="Step 4/5: Simulating Final Feedback Generation...")
    log_content += "Step 4/5: Final Feedback generation simulation complete.\n"
    yield (gr.update(), gr.update(), "Step 4/5: Simulating Final Feedback Generation...", gr.update(), gr.update(), gr.update(), final_feedback_md, gr.update(), log_content)
    time.sleep(4)
    progress(1.0, desc="Step 5/5: Simulating MCQ Generation...")
    log_content += "Step 5/5: MCQ generation simulation complete.\n"
    yield (gr.update(), gr.update(), "Step 5/5: Simulating MCQ Generation...", gr.update(), gr.update(), gr.update(), gr.update(), mcqs_md, log_content)
    log_content += "\nDemo simulation completed successfully."
    yield (
        gr.Button(interactive=True, value="🚀 Run Single Case"),
        gr.Button(interactive=True, value="Run All Cases"),
        gr.Textbox(value="Demo simulation complete.", interactive=False),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        log_content
    )

def run_pipeline_and_stream_logs(case_id, run_all, s2_api_key, pubmed_api_key, progress=gr.Progress(track_tqdm=True), *all_config_args):
    if DEMO_MODE_ENABLED:
        if run_all:
            gr.Info("In Demo Mode, 'Run All Cases' is disabled. Starting simulation for the selected case.")
        yield from simulate_pipeline_run(case_id, progress, *all_config_args)
        return
    yield (
        gr.Button(interactive=False, value="Running..."),
        gr.Button(interactive=False, value="Running..."),
        gr.Textbox(value="Saving configuration...", interactive=False),
        "", "", "", "", "", ""
    )
    runtime_config = _assemble_config_from_ui(*all_config_args)
    TEMP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEMP_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(runtime_config, f, indent=2, ensure_ascii=False)
    env = os.environ.copy()
    if s2_api_key: env["SEMANTIC_SCHOLAR_API_KEY"] = s2_api_key
    if pubmed_api_key: env["PUBMED_API_KEY"] = pubmed_api_key
    command = ["python", "async_main.py", "--config", str(TEMP_CONFIG_PATH), "--debug"]
    if not run_all:
        if not case_id:
            gr.Warning("No case selected for a single run.")
            yield (gr.Button(interactive=True, value="🚀 Run Single Case"), gr.Button(interactive=True, value="Run All Cases"), gr.Textbox(value="Error: No case selected.", interactive=False), "", "", "", "", "", "")
            return
        command.extend(["--case_id", case_id])
    status_msg = "Processing all cases in batch..." if run_all else f"Processing single case: {case_id}"
    progress(0, desc=status_msg)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1, env=env)
    total_log_content = ""
    if process.stdout:
        for line in iter(process.stdout.readline, ''):
            total_log_content += line
            yield (gr.update(), gr.update(), gr.Textbox(value=status_msg, interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), total_log_content)
    process.wait()
    progress(1, desc="Completed")
    if process.returncode != 0:
        error_msg = f"Pipeline process failed with exit code {process.returncode}."
        gr.Error(error_msg)
        total_log_content += f"\n\n--- ERROR ---\n{error_msg}\n"
        yield (gr.Button(interactive=True, value="🚀 Run Single Case"), gr.Button(interactive=True, value="Run All Cases"), gr.Textbox(value="Pipeline Failed", interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), total_log_content)
        return
    final_status_msg = "Batch processing complete." if run_all else f"Processing complete for case {case_id}."
    display_case_id = case_id
    yield (gr.update(), gr.update(), gr.Textbox(value=f"{final_status_msg} Parsing results...", interactive=False), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), total_log_content)
    input_md, retrieval_md, summary_md, final_feedback_md, mcqs_md, status = parse_and_format_output(runtime_config, display_case_id)
    if status != "OK":
        gr.Error(f"Failed to parse results for case {display_case_id}: {status}")
        final_status_msg += f" Failed to display results for '{display_case_id}'."
    else:
        final_status_msg += f" Displaying results for '{display_case_id}'."
    yield (
        gr.Button(interactive=True, value="🚀 Run Single Case"),
        gr.Button(interactive=True, value="Run All Cases"),
        gr.Textbox(value=final_status_msg, interactive=False),
        input_md, retrieval_md, summary_md, final_feedback_md, mcqs_md,
        total_log_content
    )

# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Default(), css=CUSTOM_CSS, title="Medical RAG Pipeline Demo") as demo:
    initial_config = setup_and_load_initial_config()
    load_cases_from_file(initial_config.get("keyword_file", ""))

    gr.Markdown("""
    <div style='text-align: center;'>
      <h1 style='font-size: 32px; font-weight: 700;'>MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education</h1>
    </div>
    """)
    gr.Image(value="assets/figure1.png", show_label=False, container=False, interactive=False, height=100)
    gr.Markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
      <div style='display: flex; justify-content: center; align-items: center; gap: 5px;'>
        <a href='https://github.com/yale-nlp/medical-rag' target='_blank'><img src='https://img.shields.io/badge/GitHub-Repo-blue?logo=github&style=for-the-badge'></a>
        <a href='https://2025.emnlp.org/' target='_blank'><img src='https://img.shields.io/badge/EMNLP-2025-orange?style=for-the-badge'></a>
        <a href='https://arxiv.org/abs/your-paper-id' target='_blank'><img src='https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b.svg?style=for-the-badge'></a>
      </div>
    </div>
    """)

    with gr.Tabs():
        with gr.TabItem("🚀 Pipeline Explorer"):
            with gr.Row():
                case_selector = gr.Dropdown(label="Select a Case", choices=[c.get("case_id") for c in ALL_CASES], value=ALL_CASES[0].get("case_id") if ALL_CASES else None, scale=3)
                with gr.Column(scale=2, min_width=250):
                    run_button = gr.Button("🚀 Run Single Case", variant="primary")
                    run_all_button = gr.Button("Run All Cases")
            
            status_update = gr.Textbox(label="Status", interactive=False, placeholder="Pipeline status will be shown here...")
            
            with gr.Column():
                with gr.Accordion("Step 1: Input (Report & Pre-defined Keywords)", open=True, elem_classes=["section-header"]):
                    output_input = gr.Markdown()
                with gr.Accordion("Step 2: Retrieval & Reranking", open=True, elem_classes=["section-header"]):
                    output_retrieval = gr.Markdown()
                with gr.Accordion("Step 3: Textbook Summaries", open=True, elem_classes=["section-header"]):
                    output_summary = gr.Markdown()
                with gr.Accordion("Step 4: Final Feedback", open=True, elem_classes=["section-header"]):
                    output_final_feedback = gr.Markdown()
                with gr.Accordion("Step 5: Multi-Choice Questions", open=True, elem_classes=["section-header"]):
                    output_mcqs = gr.Markdown()

        with gr.TabItem("⚙️ Configuration"):
            gr.Markdown("## System Configuration")
            gr.Markdown("Modify settings here. The pipeline will use these values when you click 'Run Pipeline' in the next tab.")
            with gr.Accordion("📁 I/O Settings", open=True, elem_classes=["section-header"]):
                cfg_keyword_file = gr.Textbox(label="Keyword Input File Path", value=initial_config.get("keyword_file"))
                cfg_feedback_dir = gr.Textbox(label="Feedback Output Directory", value=initial_config.get("feedback_dir"))
                cfg_s2_retrieved_file = gr.Textbox(label="Prefetched Semantic Scholar File (Optional)", value=initial_config.get("s2_retrieved_file"))
                cfg_pubmed_retrieved_file = gr.Textbox(label="Prefetched PubMed File (Optional)", value=initial_config.get("pubmed_retrieved_file"))

            with gr.Accordion("🌐 Retrieval Settings", open=True, elem_classes=["section-header"]):
                cfg_retrievers = gr.CheckboxGroup(label="Retrievers to Use", choices=["local", "semantic", "pubmed"], value=initial_config.get("retrievers"))
                cfg_local_index_path = gr.Textbox(label="Local Index Path", value=initial_config.get("local_index_path"))
                cfg_hf_embedding_model = gr.Textbox(label="HuggingFace Embedding Model", value=initial_config.get("hf_embedding_model"))
                cfg_s2_api_key = gr.Textbox(label="Semantic Scholar API Key", value=os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""), type="password")
                cfg_pubmed_api_key = gr.Textbox(label="PubMed API Key", value=os.getenv("PUBMED_API_KEY", ""), type="password")

            with gr.Accordion("🧠 Reranker Service", open=False, elem_classes=["section-header"]):
                cfg_reranker_model = gr.Textbox(label="Model Path", value=initial_config.get("services", {}).get("reranker_service", {}).get("model"))
                with gr.Row():
                    cfg_reranker_gpu = gr.Textbox(label="GPU ID(s)", value=initial_config.get("services", {}).get("reranker_service", {}).get("gpu_id"))
                    cfg_reranker_tp = gr.Slider(label="Tensor Parallel Size", minimum=1, maximum=8, step=1, value=initial_config.get("services", {}).get("reranker_service", {}).get("tensor_parallel_size", 1))
                    cfg_rerank_k = gr.Slider(label="Rerank Top-K", minimum=1, maximum=10, step=1, value=initial_config.get("services", {}).get("reranker_service", {}).get("rerank_candidates_num", 2))
                cfg_reranker_instr = gr.Textbox(label="Reranker Instruction", lines=2, value=initial_config.get("services", {}).get("reranker_service", {}).get("qwen_reranker_params", {}).get("qwen_instruction_task"))
            with gr.Accordion("✍️ Generation Service", open=False, elem_classes=["section-header"]):
                cfg_gen_model = gr.Textbox(label="Model Path", value=initial_config.get("services", {}).get("generation_service", {}).get("model"))
                with gr.Row():
                    cfg_gen_gpu = gr.Textbox(label="GPU ID(s)", value=initial_config.get("services", {}).get("generation_service", {}).get("gpu_id"))
                    cfg_gen_tp = gr.Slider(label="Tensor Parallel Size", minimum=1, maximum=8, step=1, value=initial_config.get("services", {}).get("generation_service", {}).get("tensor_parallel_size", 1))
            with gr.Accordion("🛠️ Generation Parameters", open=False, elem_classes=["section-header"]):
                with gr.Tabs():
                    with gr.TabItem("Textbook Summary"):
                        cfg_ts_max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=8192, step=1, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("textbook_summary",{}).get("max_tokens", 512))
                        cfg_ts_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("textbook_summary",{}).get("temperature", 0.1))
                        cfg_ts_top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("textbook_summary",{}).get("top_p", 0.9))
                    with gr.TabItem("Final Report"):
                        cfg_ff_max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=8192, step=1, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("final_report",{}).get("max_tokens", 2048))
                        cfg_ff_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("final_report",{}).get("temperature", 0.2))
                        cfg_ff_top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("final_report",{}).get("top_p", 0.9))
                    with gr.TabItem("MCQ Generation"):
                        cfg_mcq_max_tokens = gr.Slider(label="Max Tokens", minimum=1, maximum=8192, step=1, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("mcq_generation",{}).get("max_tokens", 1024))
                        cfg_mcq_temp = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("mcq_generation",{}).get("temperature", 0.2))
                        cfg_mcq_top_p = gr.Slider(label="Top-P", minimum=0.0, maximum=1.0, step=0.05, value=initial_config.get("services",{}).get("generation_service",{}).get("generation_params",{}).get("mcq_generation",{}).get("top_p", 0.9))
            with gr.Accordion("📜 System Prompts", open=False, elem_classes=["section-header"]):
                with gr.Tabs():
                    with gr.TabItem("Textbook Summary"):
                        cfg_ts_persona = gr.Textbox(label="System Persona", lines=3, value=initial_config.get("system", {}).get("textbook_summary", {}).get("system_persona"))
                        cfg_ts_template = gr.Textbox(label="User Instruction Template", lines=7, value=initial_config.get("system", {}).get("textbook_summary", {}).get("user_instruction_template"))
                    with gr.TabItem("Final Feedback"):
                        cfg_ff_persona = gr.Textbox(label="System Persona", lines=3, value=initial_config.get("system", {}).get("final_feedback", {}).get("system_persona"))
                        cfg_ff_template = gr.Textbox(label="User Instruction Template", lines=7, value=initial_config.get("system", {}).get("final_feedback", {}).get("user_instruction_template"))
                    with gr.TabItem("Questioner (MCQ)"):
                        cfg_q_persona = gr.Textbox(label="System Persona", lines=3, value=initial_config.get("system", {}).get("questioner", {}).get("system_persona"))
                        cfg_q_template = gr.Textbox(label="User Instruction Template", lines=7, value=initial_config.get("system", {}).get("questioner", {}).get("user_instruction_template"))

        with gr.TabItem("📜 Terminal Logs"):
            log_output = gr.Textbox(lines=30, interactive=False, autoscroll=True, label="Pipeline Logs")

    all_config_components = [
        cfg_keyword_file, cfg_feedback_dir, cfg_hf_embedding_model, cfg_retrievers, cfg_local_index_path,
        cfg_s2_retrieved_file, cfg_pubmed_retrieved_file,
        cfg_reranker_model, cfg_reranker_gpu, cfg_reranker_tp, cfg_rerank_k, cfg_reranker_instr,
        cfg_gen_model, cfg_gen_gpu, cfg_gen_tp,
        cfg_ts_max_tokens, cfg_ts_temp, cfg_ts_top_p,
        cfg_ff_max_tokens, cfg_ff_temp, cfg_ff_top_p,
        cfg_mcq_max_tokens, cfg_mcq_temp, cfg_mcq_top_p,
        cfg_ts_persona, cfg_ts_template,
        cfg_ff_persona, cfg_ff_template,
        cfg_q_persona, cfg_q_template
    ]
    
    api_key_components = [cfg_s2_api_key, cfg_pubmed_api_key]
    
    output_components = [
        run_button, run_all_button, status_update,
        output_input, output_retrieval, output_summary, output_final_feedback, output_mcqs, log_output
    ]
    
    cfg_keyword_file.blur(fn=load_cases_from_file, inputs=[cfg_keyword_file], outputs=None).then(fn=update_case_dropdown, inputs=None, outputs=[case_selector])
    
    run_button.click(
        fn=run_pipeline_and_stream_logs, 
        inputs=[case_selector, gr.Checkbox(value=False, visible=False)] + api_key_components + all_config_components, 
        outputs=output_components
    )
    
    run_all_button.click(
        fn=run_pipeline_and_stream_logs, 
        inputs=[case_selector, gr.Checkbox(value=True, visible=False)] + api_key_components + all_config_components, 
        outputs=output_components
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gradio UI for the Medical RAG Pipeline.")
    parser.add_argument("--demo_mode", action="store_true", help="Run the UI in demonstration mode, which simulates pipeline execution using pre-generated results.")
    args = parser.parse_args()
    
    DEMO_MODE_ENABLED = args.demo_mode

    if TEMP_CONFIG_PATH.exists():
        os.remove(TEMP_CONFIG_PATH)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [GradioApp] - %(message)s')
    
    if DEMO_MODE_ENABLED:
        print("\n" + "="*50)
        print("🚀 RUNNING IN DEMO MODE 🚀")
        print(f"Will use static result file: '{DEMO_RESULT_FILE_PATH}'")
        print("Pipeline execution will be simulated.")
        print("="*50 + "\n")

    demo.launch(share=True, allowed_paths=["assets"])