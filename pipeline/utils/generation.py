import asyncio
import logging
from typing import Dict, Any, List
from llm_services import LocalLLMService
from utils.qwen_parsing import extract_qwen_response

async def generate_textbook_summaries_async(
    llm_service: LocalLLMService,
    local_pages_per_keyword: Dict[str, List[Dict]],
    generation_config: Dict[str, Any],
    system_prompts: Dict[str, Any]
) -> Dict[str, str]:
    """Generates textbook summaries for all keywords in parallel using asyncio."""
    prompt_cfg = system_prompts.get("textbook_summary", {})
    settings = generation_config.get("generation_params", {}).get("textbook_summary", {})
    sys_persona = prompt_cfg.get("system_persona", "You are a helpful assistant.")
    template = prompt_cfg.get("user_instruction_template")
    if not template:
        logging.error("Textbook summary user instruction template not found in config.")
        return {kw: "Config Error: Template not found." for kw in local_pages_per_keyword}
    tasks = {}
    logging.debug(f"Creating {len(local_pages_per_keyword)} textbook summary generation tasks...")
    for kw, pages in local_pages_per_keyword.items():
        if pages and isinstance(pages, list):
            pages_block = "\n\n".join([f"Page {p.get('index', 'N/A')}:\n{p.get('text', '')}" for p in pages])
            if pages_block.strip():
                full_user_prompt = template.format(keyword=kw, pages_block_text=pages_block)
                tasks[kw] = llm_service.generate_text(sys_persona, full_user_prompt, settings)
            else:
                tasks[kw] = asyncio.sleep(0, result="No text content in retrieved pages.")
        else:
            tasks[kw] = asyncio.sleep(0, result="No local pages retrieved.")
    if not tasks: return {}
    raw_results = await asyncio.gather(*tasks.values())
    if "qwen" in generation_config.get("model", "").lower():
        logging.debug("Qwen model detected for textbook summary. Parsing <think> blocks.")
        clean_results = [extract_qwen_response(res) for res in raw_results]
    else:
        clean_results = list(raw_results)
    summaries = dict(zip(tasks.keys(), clean_results))
    logging.debug(f"Finished generating {len(summaries)} textbook summaries.")
    return summaries

async def generate_final_report_async(
    llm_service: LocalLLMService,
    original_report: str,
    user_block: str,
    keywords: List[str],
    generation_config: Dict[str, Any],
    system_prompts: Dict[str, Any]
) -> str:
    """Generates the final synthesized AI feedback report asynchronously."""
    logging.debug("Creating final report generation task...")
    settings = generation_config.get("generation_params", {}).get("final_report", {})
    prompt_cfg = system_prompts.get("final_feedback", {})
    sys_persona = prompt_cfg.get("system_persona")
    template = prompt_cfg.get("user_instruction_template")
    if not template or not sys_persona:
        logging.error("Final feedback prompt or persona not found in config.")
        return "Config Error: Template or persona not found."
    keywords_str = "\n- ".join(keywords)
    full_user_prompt = template.format(keywords_list_str=keywords_str, original_reviewer_report=original_report, user_block_for_final_stages=user_block)
    raw_report = await llm_service.generate_text(sys_persona, full_user_prompt, settings)
    if "qwen" in generation_config.get("model", "").lower():
        logging.debug("Qwen model detected for final report. Parsing <think> block.")
        clean_report = extract_qwen_response(raw_report)
        return clean_report
    logging.debug("Finished generating final report.")
    return raw_report

async def generate_mcqs_async(
    llm_service: LocalLLMService,
    mcq_input_context: str,
    keywords: List[str],
    generation_config: Dict[str, Any],
    system_prompts: Dict[str, Any]
) -> str:
    """Generates Multiple Choice Questions asynchronously."""
    logging.debug("Creating MCQ generation task...")
    settings = generation_config.get("generation_params", {}).get("mcq_generation", {})
    prompt_cfg = system_prompts.get("questioner", {})
    sys_persona = prompt_cfg.get("system_persona")
    template = prompt_cfg.get("user_instruction_template")
    if not template or not sys_persona:
        logging.error("Questioner prompt or persona not found in config.")
        return "Config Error: Template or persona not found."
    keywords_str = "\n- ".join(keywords)
    full_user_prompt = template.format(keywords_list_str=keywords_str, mcq_input_context=mcq_input_context)
    raw_mcqs = await llm_service.generate_text(sys_persona, full_user_prompt, settings)
    if "qwen" in generation_config.get("model", "").lower():
        logging.debug("Qwen model detected for MCQs. Parsing <think> block.")
        clean_mcqs = extract_qwen_response(raw_mcqs)
        return clean_mcqs
    logging.debug("Finished generating MCQs.")
    return raw_mcqs