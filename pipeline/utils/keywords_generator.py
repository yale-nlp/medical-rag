import os
import re
import json
import gc
import time
import torch
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import vllm

# --------------------- Environment ---------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
load_dotenv()


# ---------------------- Helpers ------------------------
def get_impression_from_report(summary: str) -> str:
    """Extract the 'Impression' section from a radiology report."""
    lower = summary.lower()
    if "impression: " in lower:
        summary = summary[lower.find("impression: ") + len("impression: "):]
    for token in ("reported by:", "reported and signed by:", "report initiated by:"):
        pos = summary.lower().find(token)
        if pos != -1:
            summary = summary[:pos]
            break
    return summary.strip()


def choose_model() -> str:
    """Prompt user to choose OpenAI or vLLM."""
    print("Select keyword generation model:")
    print("1: OpenAI GPT‑4o‑mini")
    print("2: HuggingFace Llama‑3 (vLLM)")
    return "openai" if input("Enter 1 or 2: ").strip() == "1" else "vllm"


# ------------------- vLLM Wrappers --------------------
def build_llm():
    return vllm.LLM(
        model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        max_model_len=2048,
    )


def llama_generate(prompt: str, llm, max_tokens: int = 128) -> str:
    """
    Run vLLM with deterministic sampling.
    Stop as soon as the JSON object is closed ('}').
    """
    from vllm import SamplingParams
    params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        repetition_penalty=1.05,
        stop=["}\n"],
        max_tokens=max_tokens
    )
    text = llm.generate([prompt], params)[0].outputs[0].text
    if "}" in text:
        return text[: text.rfind("}") + 1 ].strip()
    return text.strip()


def free_llm(llm):
    """Release GPU memory used by vLLM."""
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)


# ------------------- JSON Parsing ---------------------
CODEBLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.I)


def extract_json_openai(resp: str) -> list[str]:
    """
    Parse OpenAI output by first looking for a ```json``` block, else
    assume the entire output is JSON.
    """
    m = CODEBLOCK.search(resp)
    if m:
        content = m.group(1)
        obj = json.loads(content)
    else:
        obj = json.loads(resp)
    if not isinstance(obj, dict) or "keywords" not in obj:
        raise ValueError("JSON missing 'keywords' key")
    if not isinstance(obj["keywords"], list):
        raise ValueError("'keywords' is not a list")
    return obj["keywords"]


def extract_json_llama(resp: str) -> list[str]:
    """
    Parse Llama output by first stripping any ```json``` block,
    else falling back to first {…} snippet.
    """
    # try code block
    m = CODEBLOCK.search(resp)
    if m:
        frag = m.group(1)
    else:
        start = resp.find('{')
        end   = resp.rfind('}') + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        frag = resp[start:end]
    obj = json.loads(frag)
    if isinstance(obj, dict):
        kws = obj.get("keywords", [])
    elif isinstance(obj, list):
        kws = obj
    else:
        raise ValueError("Parsed JSON is not dict or list")
    if not isinstance(kws, list):
        raise ValueError("'keywords' is not a list")
    return kws


# -------------------- Main Pipeline -------------------
def generate_keywords(in_file: str, out_file: str):
    choice = choose_model()

    # load prompts
    curr_script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(curr_script_dir, "../configs/prompts.json"), "r", encoding="utf-8") as f:
        prompts = json.load(f)
    openai_prompt = prompts["system"]["analyzer"]
    llama_prompt = prompts["system"]["keyword_llama"]

    # read input
    if in_file.lower().endswith(".csv"):
        df = pd.read_csv(in_file, index_col=0)
        reports = []
        for case_id, row in df.iterrows():
            reports.append({
                "Unnamed: 0": case_id,
                "reviewer_report": row["reviewer_report"],
                "keywords": []
            })
    else:
        with open(in_file, "r", encoding="utf-8") as f:
            reports = json.load(f)

    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if choice == "openai" else None
    llama = build_llm() if choice == "vllm" else None

    # Llama 3 special tokens
    BOS = "<|begin_of_text|>"
    EOS = "<|eot_id|>"
    SH  = "<|start_header_id|>"
    EH  = "<|end_header_id|>"

    for idx, rpt in enumerate(reports):
        full = rpt.get("reviewer_report", "")
        impr = get_impression_from_report(full)
        user_section = f"Final_report: {full}\nImpression: {impr}"

        if choice == "openai":
            resp = openai_client.responses.create(
                model="gpt-4o-mini",
                instructions=openai_prompt,
                input=user_section
            ).output_text.strip()
        else:
            prompt = (
                f"{BOS}{SH}system{EH}\n"
                f"{llama_prompt}{EOS}"
                f"{SH}user{EH}\n"
                f"{user_section}{EOS}"
                f"{SH}assistant{EH}"
            )
            resp = llama_generate(prompt, llama)

        rpt["_raw_response"] = resp
        print(f"\n[RAW {idx}]\n{resp}\n")

        try:
            if choice == "openai":
                kws = extract_json_openai(resp)
            else:
                kws = extract_json_llama(resp)
            rpt["keywords"] = kws
        except Exception as e:
            print(f"[{idx}] parse failed: {e}")
            rpt["keywords"] = []

        print(f"[{idx}] => {rpt['keywords']}")

    if llama:
        free_llm(llama)

    # write out
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=4)


# ---- Entry Point ----
if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    generate_keywords(
        os.path.join(curr_dir, "../data/report_sample.csv"),
        os.path.join(curr_dir, "../data/keywords_sample.json")
    )