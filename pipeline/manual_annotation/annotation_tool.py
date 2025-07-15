import streamlit as st
import re
from pathlib import Path
from datetime import datetime
import json
import copy
import os
import time

# ───────────────────────── 1. Configuration ───────────────────────── #
st.set_page_config(page_title="Medical-RAG Annotation", layout="wide")

# --- File Paths & Experiment Definitions ---
BASE_RESULT_DIR = Path("./result")
BACKUP_DIR = BASE_RESULT_DIR / "annotated_backups"
IN_PROGRESS_DIR = Path("./result/annotation_in_progress")
IN_PROGRESS_DIR.mkdir(exist_ok=True, parents=True)
BACKUP_DIR.mkdir(exist_ok=True, parents=True)

def get_in_progress_path(annotator_name: str) -> Path:
    if not annotator_name:
        return None
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', annotator_name)
    return IN_PROGRESS_DIR / f"in_progress_{safe_name}.json"

EXPERIMENTS = {
    "exp1": {
        "name": "MEDGEMMA",
        "input_file": BASE_RESULT_DIR / "final_sample_250/cross25add_medgemma-27b_part_A.json",
    },
    "exp2": {
        "name": "LLAMA",
        "input_file": BASE_RESULT_DIR / "final_sample_250/cross25add_llama3.3-70b_part_A.json",
    }
}

# --- Annotation Criteria ---
ANNOTATION_CRITERIA = {
    "keyword_appropriateness": (1, 5, "How appropriate is this specific keyword for the original report?"),
    "suggested_keywords": ("text", "Suggest better or additional keywords here. Separate with commas."),
    "paper_relevance": (1, 5, "How relevant is this specific paper to the clinical context of the original report?"),
    "summary_quality": (1, 5, "Evaluate the textbook summary's accuracy, factuality, and educational value."),
    "final_feedback_quality": (1, 5, "Evaluate the final feedback's clinical utility, accuracy, and educational value against the *original report*."),
    "mcq_quality": (1, 5, "Evaluate this specific MCQ's relevance, correctness, and quality of distractors."),
    "overall_correctness": {
        "label": "Assess the overall clinical correctness of all generated content.",
        "options": {
            1: "No error",
            2: "Minor error (trivial)",
            3: "Major error (clinically significant)"
        }
    },
    "overall_comments": ("text", "Provide any general comments about this model's output for the case.")
}

# --- UI Constants ---
RATING_OPTIONS = [1, 2, 3, 4, 5]
ANNOTATION_SECTIONS = ["Keywords", "Papers", "Summaries", "MCQs", "Feedback", "Overall"]

# ───────────────────────── 2. Data & State Functions ───────────────────────── #

# UPDATE: Removed @st.cache_data decorator to fix the session resume bug.
# This function MUST check the filesystem every time.
def load_and_prepare_data(exp_configs, annotator_name):
    """Loads data, merges by case_id, handles resuming sessions for a specific annotator, and sorts naturally."""
    in_progress_file = get_in_progress_path(annotator_name)
    if in_progress_file and in_progress_file.exists():
        st.sidebar.info(f"Resuming session for **{annotator_name}**.")
        with open(in_progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    exp_data = {}
    for exp_id, config in exp_configs.items():
        try:
            with open(config["input_file"], 'r', encoding='utf-8') as f:
                data = json.load(f)
                case_list = data.get("all_processed_reports", [])
                exp_data[exp_id] = {case['case_id']: case for case in case_list}
        except FileNotFoundError:
            st.error(f"Error: Input file for {config['name']} not found at '{config['input_file']}'.")
            return []
        except (json.JSONDecodeError, TypeError) as e:
            st.error(f"Error: Failed to parse JSON for {config['name']}. Details: {e}")
            return []

    all_case_ids = set()
    for data_dict in exp_data.values():
        all_case_ids.update(data_dict.keys())

    if not all_case_ids:
        st.warning("No cases found in the provided files.")
        return []

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    sorted_case_ids = sorted(list(all_case_ids), key=natural_sort_key)

    comparison_data = []
    for case_id in sorted_case_ids:
        comparison_data.append({
            "case_id": case_id,
            "exp1_data": exp_data.get("exp1", {}).get(case_id),
            "exp2_data": exp_data.get("exp2", {}).get(case_id),
            "annotations": {}
        })

    return comparison_data

def initialize_annotations(case_data):
    annotations = case_data.get("annotations", {})
    annotations.setdefault("skipped", False)
    annotations.setdefault("skip_reason", "")
    annotations.setdefault("keyword_appropriateness", {})
    annotations.setdefault("suggested_keywords", "")
    annotations.setdefault("paper_relevance", {})
    for exp_id in EXPERIMENTS.keys():
        annotations.setdefault(f"{exp_id}_summary_quality", 0)
        annotations.setdefault(f"{exp_id}_final_feedback_quality", 0)
        annotations.setdefault(f"{exp_id}_mcq_quality", [])
        annotations.setdefault(f"{exp_id}_overall_correctness", 0)
        annotations.setdefault(f"{exp_id}_overall_comments", "")
    case_data["annotations"] = annotations

def is_case_annotated(annotations: dict) -> bool:
    """Checks if ANY annotation has been made (for export purposes)."""
    if annotations.get("skipped", False): return True
    return any(annotations.get("keyword_appropriateness")) or \
           any(annotations.get("paper_relevance")) or \
           any(annotations.get(f"{exp_id}_summary_quality", 0) for exp_id in EXPERIMENTS.keys()) or \
           any(annotations.get(f"{exp_id}_final_feedback_quality", 0) for exp_id in EXPERIMENTS.keys()) or \
           any(any(rating for rating in annotations.get(f"{exp_id}_mcq_quality", [])) for exp_id in EXPERIMENTS.keys())

def check_section_completion(section_name, annotations, current_case):
    """Checks if a SPECIFIC section is fully annotated."""
    if section_name == "Keywords":
        report_source_data = current_case.get("exp1_data") or current_case.get("exp2_data") or {}
        keywords_to_rate = report_source_data.get("original_keywords", [])
        if not keywords_to_rate:
            return True
        rated_keywords = annotations.get("keyword_appropriateness", {})
        return all(kw in rated_keywords for kw in keywords_to_rate)
    
    if section_name == "Papers":
        exp1_papers = [p for papers in current_case.get("exp1_data", {}).get("evidence_reranked_papers", {}).values() for p in papers]
        exp2_papers = [p for papers in current_case.get("exp2_data", {}).get("evidence_reranked_papers", {}).values() for p in papers]
        all_papers = exp1_papers + exp2_papers
        unique_papers = list({p['url']: p for p in all_papers if p.get('url')}.values())
        if not unique_papers: return True
        return all(p.get('url') in annotations.get("paper_relevance", {}) for p in unique_papers)

    if section_name == "Summaries":
        for exp_id in EXPERIMENTS.keys():
            model_data = current_case.get(f"{exp_id}_data")
            if model_data and model_data.get("generated_textbook_summaries"):
                if annotations.get(f"{exp_id}_summary_quality", 0) == 0:
                    return False
        return True

    if section_name == "Feedback":
        for exp_id in EXPERIMENTS.keys():
            model_data = current_case.get(f"{exp_id}_data")
            if model_data and model_data.get("generated_final_feedback"):
                if annotations.get(f"{exp_id}_final_feedback_quality", 0) == 0:
                    return False
        return True

    if section_name == "MCQs":
        for exp_id in EXPERIMENTS.keys():
            model_data = current_case.get(f"{exp_id}_data")
            if not model_data: continue
            mcqs_text = model_data.get("generated_mcqs", "")
            if not mcqs_text: continue
            mcq_matches = list(re.finditer(r"####\s*(?P<keyword>.*?)\n(?P<mcq_block>[\s\S]*?)(?=####|$)", mcqs_text))
            num_total_mcqs = sum(len(parse_mcq_block(m.group('mcq_block'))) for m in mcq_matches)
            if num_total_mcqs > 0:
                ratings = annotations.get(f"{exp_id}_mcq_quality", [])
                if len(ratings) != num_total_mcqs or any(r == 0 for r in ratings):
                    return False
        return True
        
    if section_name == "Overall":
        for exp_id in EXPERIMENTS.keys():
            if annotations.get(f"{exp_id}_overall_correctness", 0) == 0:
                return False
        return True
    
    return False

def is_case_FULLY_annotated(annotations, current_case):
    """Checks if ALL sections for a case are fully annotated."""
    if annotations.get("skipped", False):
        return True
    for section in ANNOTATION_SECTIONS:
        if not check_section_completion(section, annotations, current_case):
            return False
    return True

def save_in_progress_data(all_cases_data, annotator_name):
    """Saves the current state to the annotator-specific file."""
    in_progress_file = get_in_progress_path(annotator_name)
    if not in_progress_file:
        st.error("Annotator name is not set. Cannot save progress.")
        return
    try:
        with open(in_progress_file, 'w', encoding='utf-8') as f:
            json.dump(all_cases_data, f, indent=2, ensure_ascii=False)
        st.toast(f"✅ Progress for {annotator_name} saved!", icon="💾")
    except Exception as e:
        st.error(f"Error saving progress: {e}")

def save_final_annotated_data(all_cases_data, annotator_name):
    """Saves the final data with the annotator's name in the filename."""
    if not annotator_name:
        st.warning("⚠️ Please enter an annotator name before exporting.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    for exp_id, exp_config in EXPERIMENTS.items():
        annotated_cases_for_exp = []
        for case in all_cases_data:
            if is_case_annotated(case['annotations']) and not case['annotations'].get('skipped', False) and case.get(f"{exp_id}_data"):
                new_case_record = copy.deepcopy(case[f"{exp_id}_data"])
                annotation_block = {"annotator": annotator_name, "annotation_timestamp": datetime.now().isoformat()}
                annotation_block["keyword_appropriateness"] = case["annotations"].get("keyword_appropriateness")
                annotation_block["suggested_keywords"] = case["annotations"].get("suggested_keywords")
                annotation_block["paper_relevance"] = case["annotations"].get("paper_relevance")
                for key, value in case["annotations"].items():
                    if key.startswith(exp_id):
                        clean_key = key.replace(f"{exp_id}_", "")
                        annotation_block[clean_key] = value
                new_case_record["annotation"] = annotation_block
                annotated_cases_for_exp.append(new_case_record)
        if not annotated_cases_for_exp:
            st.warning(f"No annotated cases found for {exp_config['name']}. Nothing to export.")
            continue
        output_filename = BACKUP_DIR / f"{timestamp}_{annotator_name}_{exp_id}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(annotated_cases_for_exp, f, indent=2, ensure_ascii=False)
        st.success(f"✅ **{len(annotated_cases_for_exp)}** annotated cases for **{exp_config['name']}** saved to **{output_filename}**")

def parse_mcq_block(text: str) -> list:
    if not isinstance(text, str): return []
    question_blocks = re.split(r'\n(?=Q\d+\.)', text.strip())
    parsed_mcqs = []
    mcq_pattern = re.compile(r"Q\d+\.\s*(?P<stem>.*?)\s*A\.\s*(?P<opt_a>.*?)\s*B\.\s*(?P<opt_b>.*?)\s*C\.\s*(?P<opt_c>.*?)\s*D\.\s*(?P<opt_d>.*?)\s*Answer:\s*(?P<answer>.*?)\s*Explanation:\s*(?P<explanation>.*)", re.DOTALL | re.IGNORECASE)
    for block in question_blocks:
        if not block.strip(): continue
        match = mcq_pattern.search(block)
        if match:
            parsed_mcqs.append(match.groupdict())
        else:
            parsed_mcqs.append({"stem": "--- Parsing Failed ---", "raw": block})
    return parsed_mcqs

# ───────────────────────── 3. Streamlit UI Rendering ───────────────────────── #

st.sidebar.title("Annotation Control Panel")

annotator_name = st.sidebar.text_input(
    "🖊️ **Enter Your Name to Begin**", 
    value=st.session_state.get("annotator", ""), 
    key="annotator_name_input"
).strip()
st.session_state.annotator = annotator_name

if not annotator_name:
    st.warning("Please enter your name in the sidebar to load data and start annotating.")
    st.stop()

if "cases" not in st.session_state or st.session_state.get("current_annotator") != annotator_name:
    st.session_state.cases = load_and_prepare_data(EXPERIMENTS, annotator_name)
    if st.session_state.cases:
        for case in st.session_state.cases:
            initialize_annotations(case)
    st.session_state.current_annotator = annotator_name
    st.session_state.selected_index = 0

total_cases = len(st.session_state.get("cases", []))
if total_cases == 0:
    st.error("No cases loaded. Please check data file paths and format.")
    st.stop()

# --- Sidebar Controls ---
def format_case_label(idx):
    case = st.session_state.cases[idx]
    if case['annotations'].get('skipped'):
        return f"❓ Case ID: {case.get('case_id', idx)}"
    if is_case_FULLY_annotated(case['annotations'], case):
        return f"✅ Case ID: {case.get('case_id', idx)}"
    else:
        return f"❌ Case ID: {case.get('case_id', idx)}"

selected_index = st.session_state.get("selected_index", 0)
new_selected_index = st.sidebar.selectbox("Select Case", options=range(total_cases), format_func=format_case_label, index=selected_index)
if new_selected_index != selected_index:
    st.session_state.selected_index = new_selected_index
    st.rerun()

current_case = st.session_state.cases[selected_index]
annotations = current_case["annotations"]
report_source_data = current_case.get("exp1_data") or current_case.get("exp2_data") or {}

with st.sidebar.expander("📖 Original Report", expanded=True):
    st.markdown(report_source_data.get("original_reviewer_report", "N/A"))

with st.sidebar.expander("📘 Annotation Guidelines", expanded=False):
    st.markdown("""...""")

completed_count = sum(1 for case in st.session_state.cases if is_case_FULLY_annotated(case['annotations'], case))
progress_percent = (completed_count / total_cases) if total_cases > 0 else 0
st.sidebar.markdown(f"**Overall Progress: {completed_count} / {total_cases} Cases**")
st.sidebar.progress(progress_percent)

st.sidebar.divider()

if st.sidebar.button("💾 Save Progress", use_container_width=True, type="primary"):
    save_in_progress_data(st.session_state.cases, annotator_name)
if st.sidebar.button("📤 Export Final Annotated Data", use_container_width=True):
    save_in_progress_data(st.session_state.cases, annotator_name)
    save_final_annotated_data(st.session_state.cases, annotator_name)

# --- Main Content Area ---
st.header(f"Annotation for Case ID: {current_case['case_id']}")

section_labels = [f"{'✅' if check_section_completion(s, annotations, current_case) else '❌'} {s}" for s in ANNOTATION_SECTIONS]
keyword_tab, papers_tab, summaries_tab, mcqs_tab, feedback_tab, overall_tab = st.tabs(section_labels)

# --- Keywords Tab ---
with keyword_tab:
    with st.expander("✍️ Grading Rubric for Keywords", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Keyword Appropriateness**")
            keywords = report_source_data.get("original_keywords", [])
            if not keywords: st.info("No keywords to rate.")
            for kw in keywords:
                current_val = annotations["keyword_appropriateness"].get(kw, 0)
                new_val = st.radio(f"`{kw}`", RATING_OPTIONS, index=current_val-1 if current_val else None, key=f"{current_case['case_id']}_kw_{kw}", horizontal=True)
                if new_val:
                    annotations["keyword_appropriateness"][kw] = new_val
                elif kw in annotations["keyword_appropriateness"]:
                     del annotations["keyword_appropriateness"][kw]
        with c2:
            st.markdown("**Suggested Keywords**")
            annotations["suggested_keywords"] = st.text_area("Suggest new or better keywords", value=annotations.get("suggested_keywords", ""), key=f"{current_case['case_id']}_SK", help=ANNOTATION_CRITERIA["suggested_keywords"][1])
    
    st.divider()
    st.subheader("Keywords")
    st.json(report_source_data.get("original_keywords", []))
    st.subheader("Skip This Case")
    skip_reason = st.text_area("**If you need to skip this case, please provide a reason below.**", key=f"skip_reason_{current_case['case_id']}", value=annotations.get("skip_reason", ""))
    def skip_case(reason):
        annotations['skipped'] = True
        annotations['skip_reason'] = reason
        save_in_progress_data(st.session_state.cases, annotator_name)
        st.session_state.selected_index = (selected_index + 1) % total_cases
        st.toast(f"Case {current_case['case_id']} skipped.", icon="❓")
        st.rerun()
    st.button("❓ Skip this Case", on_click=skip_case, args=(skip_reason,), disabled=(not skip_reason), help="You must provide a reason in the text box above to skip this case.")

# --- Papers Tab ---
with papers_tab:
    st.subheader("Content & Rating: Retrieved Papers")
    exp1_papers = [p for papers in current_case.get("exp1_data", {}).get("evidence_reranked_papers", {}).values() for p in papers]
    exp2_papers = [p for papers in current_case.get("exp2_data", {}).get("evidence_reranked_papers", {}).values() for p in papers]
    all_papers = exp1_papers + exp2_papers
    unique_papers = list({p['url']: p for p in all_papers if p.get('url')}.values())

    if not unique_papers:
        st.warning("No reranked papers found for this case.")
    else:
        for i, paper in enumerate(unique_papers):
            paper_id = paper.get('url', f"paper_{i}")
            with st.container(border=True):
                content_col, rating_col = st.columns([4, 1])
                with content_col:
                    st.markdown(f"**{i+1}. {paper.get('title')}**")
                    st.markdown(f"_{paper.get('abstract', 'N/A')}_")
                    st.link_button(f"Go to Source ({paper.get('source', 'N/A')})", paper.get('url', ''))
                with rating_col:
                    st.markdown("**Relevance**")
                    current_val = annotations["paper_relevance"].get(paper_id, 0)
                    new_val = st.radio("Rate 1-5", RATING_OPTIONS, index=current_val-1 if current_val else None, key=f"{current_case['case_id']}_paper_{paper_id}", horizontal=True)
                    if new_val:
                        annotations["paper_relevance"][paper_id] = new_val

# --- Summaries Tab ---
with summaries_tab:
    with st.expander("✍️ Grading Rubric for Summaries", expanded=True):
        cols = st.columns(len(EXPERIMENTS))
        for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
            with cols[i]:
                st.markdown(f"**{exp_config['name']}**")
                model_data = current_case.get(f"{exp_id}_data")
                if not model_data or not model_data.get("generated_textbook_summaries"):
                    st.info("No summary to rate.")
                else:
                    rating_key = f"{exp_id}_summary_quality"
                    current_val = annotations.get(rating_key, 0)
                    new_val = st.radio("Summary Quality", RATING_OPTIONS, index=current_val-1 if current_val else None, key=f"{current_case['case_id']}_{rating_key}", horizontal=True)
                    if new_val:
                        annotations[rating_key] = new_val
    
    st.divider()
    st.subheader("Textbook Summaries")
    c1, c2 = st.columns(2, gap="large")
    for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
        col = c1 if i == 0 else c2
        with col:
            st.markdown(f"<h5 style='text-align: center;'>{exp_config['name']}</h5>", unsafe_allow_html=True)
            model_data = current_case.get(f"{exp_id}_data")
            if not model_data: st.warning("Data not available."); continue
            content = model_data.get("generated_textbook_summaries", {})
            with st.container(border=True, height=600):
                if content:
                    for kw, summary in content.items(): st.markdown(f"**{kw}**"); st.markdown(summary)
                else: st.markdown("_N/A_")

# --- MCQs Tab ---
with mcqs_tab:
    st.subheader("Content & Rating: Generated Multiple Choice Questions")
    c1, c2 = st.columns(2, gap="large")
    for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
        col = c1 if i == 0 else c2
        with col:
            st.markdown(f"<h5 style='text-align: center;'>{exp_config['name']}</h5>", unsafe_allow_html=True)
            model_data = current_case.get(f"{exp_id}_data")
            if not model_data: st.warning("Data not available."); continue
            
            content = model_data.get("generated_mcqs", "")
            with st.container(border=True, height=800):
                if not content: st.markdown("_No MCQs generated._")
                else:
                    mcq_matches = list(re.finditer(r"####\s*(?P<keyword>.*?)\n(?P<mcq_block>[\s\S]*?)(?=####|$)", content))
                    rating_key = f"{exp_id}_mcq_quality"
                    num_total_mcqs = sum(len(parse_mcq_block(m.group('mcq_block'))) for m in mcq_matches)
                    if len(annotations[rating_key]) != num_total_mcqs: annotations[rating_key] = [0] * num_total_mcqs

                    mcq_idx_counter = 0
                    for match in mcq_matches:
                        st.markdown(f"**Keyword: {match.group('keyword').strip()}**")
                        parsed_mcqs = parse_mcq_block(match.group('mcq_block'))
                        for q_idx, mcq in enumerate(parsed_mcqs):
                            if mcq['stem'] == "--- Parsing Failed ---":
                                st.warning(f"**MCQ Parsing Failed:**\n```\n{mcq['raw']}\n```")
                                mcq_idx_counter += 1
                                continue
                            
                            stem_html = f"<p style='color: #90EE90; margin-bottom: 5px;'><strong>Q{q_idx+1}. {mcq.get('stem', '')}</strong></p>"
                            options_html = f"<ul style='list-style-type: none; padding-left: 15px; margin-top: 0px;'><li>A. {mcq.get('opt_a', '')}</li><li>B. {mcq.get('opt_b', '')}</li><li>C. {mcq.get('opt_c', '')}</li><li>D. {mcq.get('opt_d', '')}</li></ul>"
                            answer_html = f"<p style='color: #ADD8E6;'><strong>Answer:</strong> {mcq.get('answer', '')}</p>"
                            explanation_html = f"<p style='color: #D3D3D3;'><strong>Explanation:</strong> {mcq.get('explanation', '')}</p>"
                            st.markdown(stem_html, unsafe_allow_html=True)
                            st.markdown(options_html, unsafe_allow_html=True)
                            st.markdown(answer_html, unsafe_allow_html=True)
                            st.markdown(explanation_html, unsafe_allow_html=True)
                            
                            current_val = annotations[rating_key][mcq_idx_counter]
                            new_val = st.radio("Rate Quality", RATING_OPTIONS, index=current_val-1 if current_val else None, key=f"{current_case['case_id']}_{rating_key}_{mcq_idx_counter}", horizontal=True, label_visibility="collapsed")
                            if new_val:
                                annotations[rating_key][mcq_idx_counter] = new_val
                            
                            st.markdown("---")
                            mcq_idx_counter += 1

# --- Feedback Tab ---
with feedback_tab:
    with st.expander("✍️ Grading Rubric for Final Feedback", expanded=True):
        cols = st.columns(len(EXPERIMENTS))
        for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
            with cols[i]:
                st.markdown(f"**{exp_config['name']}**")
                model_data = current_case.get(f"{exp_id}_data")
                if not model_data or not model_data.get("generated_final_feedback"):
                    st.info("No feedback to rate.")
                else:
                    rating_key = f"{exp_id}_final_feedback_quality"
                    current_val = annotations.get(rating_key, 0)
                    new_val = st.radio("Feedback Quality", RATING_OPTIONS, index=current_val-1 if current_val else None, key=f"{current_case['case_id']}_{rating_key}", horizontal=True)
                    if new_val:
                        annotations[rating_key] = new_val

    st.divider()
    st.subheader("Final Feedback")
    c1, c2 = st.columns(2, gap="large")
    for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
        col = c1 if i == 0 else c2
        with col:
            st.markdown(f"<h5 style='text-align: center;'>{exp_config['name']}</h5>", unsafe_allow_html=True)
            model_data = current_case.get(f"{exp_id}_data")
            if not model_data: st.warning("Data not available."); continue
            content = model_data.get("generated_final_feedback", "N/A")
            with st.container(border=True, height=600):
                st.markdown(content)

# --- Overall Tab ---
with overall_tab:
    with st.expander("✍️ Grading Rubric for Overall Assessment", expanded=True):
        cols = st.columns(len(EXPERIMENTS))
        for i, (exp_id, exp_config) in enumerate(EXPERIMENTS.items()):
            with cols[i]:
                st.markdown(f"**{exp_config['name']}**")
                
                rating_key = f"{exp_id}_overall_correctness"
                options_dict = ANNOTATION_CRITERIA["overall_correctness"]["options"]
                current_val = annotations.get(rating_key, 0)
                
                new_val = st.radio(
                    "Clinical Correctness", 
                    options=options_dict.keys(),
                    format_func=lambda x: f"{x}: {options_dict[x]}",
                    index=list(options_dict.keys()).index(current_val) if current_val in options_dict else None,
                    key=f"{current_case['case_id']}_{rating_key}"
                )
                if new_val:
                    annotations[rating_key] = new_val

                annotations[f"{exp_id}_overall_comments"] = st.text_area("Overall Comments", value=annotations.get(f"{exp_id}_overall_comments", ""), key=f"{current_case['case_id']}_{exp_id}_CM")
    
    st.divider()
    st.subheader("Overall Assessment")
    st.info("Provide your overall assessment for both models in the Rubric panel above.")
    
    # UPDATE: Add conditional "Go to Next Case" button
    if check_section_completion("Overall", annotations, current_case):
        st.success("This case is fully annotated!")
        if st.button("➡️ Go to Next Case", use_container_width=True, type="primary"):
            # UPDATE: Save progress before moving to the next case
            save_in_progress_data(st.session_state.cases, annotator_name)
            next_index = (selected_index + 1) % total_cases
            st.session_state.selected_index = next_index
            st.rerun()