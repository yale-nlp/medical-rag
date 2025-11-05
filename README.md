# MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: ODC-By-1.0](https://img.shields.io/badge/License-ODC--By--1.0-blue.svg)](https://opendatacommons.org/licenses/by/1-0/)
[![Paper](https://img.shields.io/badge/Paper-EMNLP_2025_Demo-yellow.svg)](https://aclanthology.org/2025.emnlp-demos.24/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/yale-nlp/MedTutor)

MedTutor is a scalable, retrieval-augmented generation (RAG) pipeline for case‑based medical education. It combines hybrid retrieval (local knowledge base + live literature search), reranking, and high‑throughput LLM generation to synthesize evidence and produce educational outputs such as feedback and multiple‑choice questions (MCQs). The system is built for speed and reproducibility with vLLM, asyncio, and multi‑GPU support.

<p align="center">
  <img src="./pipeline/assets/figure1.png" alt="MedTutor pipeline overview" width="720"/>
  <br>
  <em>Figure 1. Overview of the MedTutor pipeline.</em>
</p>

## Highlights

- High‑throughput inference with vLLM (continuous batching, multi‑GPU, tensor parallel).
- Hybrid retrieval: dense local index + live PubMed/Semantic Scholar APIs (optional).
- Reranking via cross‑encoder or local Qwen‑Reranker served through vLLM.
- Modular multi‑stage generation: textbook summaries → final feedback → MCQs.
- Configuration‑driven experiments via JSON; swap models/prompts without code changes.
- Gradio UI for exploring a single case and running the pipeline interactively.

## Repository Structure

- `pipeline/async_main.py`: Batch pipeline entrypoint (staged retrieval → generation → write results).
- `pipeline/app.py`: Gradio UI (single‑case run, live logs, optional demo mode).
- `pipeline/utils/`: Retrieval (local + live), reranker, vLLM handler, embeddings, keyword generator.
- `pipeline/configs/`: Example configs (`configs_all.json`, `gradio_config.json`, `configs_template.json`).
- `pipeline/llm_annotation/`: LLM‑as‑a‑Judge tools and configs.
- `pipeline/manual_annotation/`: Streamlit UI for human annotations.
- `pipeline/assets/`: Figures and screenshots.

## Requirements

- Python 3.11+
- Linux/macOS, CUDA GPU recommended (70B models require multiple GPUs or tensor parallelism)
- vLLM installed separately (not pinned in `requirements.txt` due to CUDA/driver variability)

## Installation

1) Create environment and install dependencies

```bash
python -m venv .venv && source .venv/bin/activate  # or conda
pip install -r requirements.txt
```

2) Install vLLM (follow official instructions for your platform/CUDA)

- vLLM docs: https://docs.vllm.ai/en/stable/getting_started/installation/

3) (Optional) Set API keys for live retrieval (place a `.env` file in `pipeline/` or export as env vars)

```
SEMANTIC_SCHOLAR_API_KEY=...
PUBMED_API_KEY=...
```

If you do not set keys, set `retrievers` to only `"local"` in the config.

## Quickstart

1) Prepare a config

```bash
cd pipeline
cp configs/configs_template.json configs/configs.json
# Edit configs/configs.json → update absolute paths: "keyword_file", "local_index_path", "feedback_dir".
# Optionally set models/GPUs under "services" and the stage "service_map".
```

2) (Optional) Build a local embedding index for textbook pages

Place your source pages under `pipeline/data/ocr_batches/` (or adjust the script), then run:

```bash
python utils/embed.py
```

This writes an embedded page file (e.g., `pipeline/data/embedded_pages_*.json`). Point `local_index_path` at this file.

3) Run the pipeline

```bash
# from pipeline/
python async_main.py --config configs/configs.json
```

Useful flags:
- `--debug` for verbose logs
- `--case_id <ID>` to process a single case from the keyword file

4) Explore the UI

```bash
# from pipeline/
python app.py
```

Demo mode is available via `--demo_mode`, but requires a local demo results file; see `app.py` for path details.

## Data Preparation

You need two inputs: (1) a keyword file per case, and (2) a local embedding index of textbook pages.

1) Keyword generation (optional)

The high‑throughput keyword generator uses a local vLLM server.

```bash
# from pipeline/
python utils/keywords_generator.py \
  --input data/keywords_sample/keyword_generator_input_sample.jsonl \
  --output data/keywords_sample/keyword_generator_output_sample.jsonl \
  --prompts configs/keywords_generator_prompt.json
```

Each output line will include `_raw_response` and a `keywords` list. Use this file as `keyword_file` in your config.

2) Local vector index

Run `python utils/embed.py` (see Quickstart) to build an index from your pages and set `local_index_path` accordingly. If you only want to test the pipeline structure, you can temporarily set `retrievers` to `["local"]` and use a small index.

### Released Datasets (Hugging Face)

- Dataset: https://huggingface.co/datasets/yale-nlp/MedTutor
- Composition (per case):
  - Inputs: `case_id`, `reviewer_report`, `keywords`
  - Retrieval: `reranked_papers_per_keyword` (title, abstract, url, source, rerank_score), `local_pages_per_keyword` (index, text)
  - Generated: `generated_textbook_summaries` (per keyword), `generated_final_feedback`, `generated_mcqs`

## Configuration Notes

- `hf_embedding_model` / `embedding_model_device`: Model/device for query embeddings (CPU works for small jobs).
- `retrievers`: Enable sources among `"local"`, `"pubmed"`, `"semantic"`.
- `keyword_file`, `local_index_path`: Absolute or project‑relative paths to your data.
- `services`: Define model workers (e.g., reranker and generator) with `gpu_id`, `tensor_parallel_size`, and per‑stage generation params.
- `service_map`: Map stages (`reranker`, `textbook_summary`, `final_report`, `mcq_generation`) to a service name.
- `system`: System personas and user templates per stage.

See `pipeline/configs/configs_all.json` for a larger 70B example and `pipeline/configs/gradio_config.json` for the UI defaults.

## Output Format

The pipeline writes a timestamped JSON under `feedback_dir`, including the resolved config and a list of processed cases. Each case includes original inputs, retrieved/reranked evidence, intermediate textbook summaries, final feedback text, and MCQs.

## Evaluation & Annotation

- LLM‑as‑a‑Judge (automated, multi‑provider): see pipeline docs at [pipeline/llm_annotation/README.md](pipeline/llm_annotation/README.md)
- Human annotation UI (Streamlit): see [pipeline/manual_annotation/README.md](pipeline/manual_annotation/README.md)

## Reproducibility

- Determinism across large LLMs can vary by hardware/driver. We recommend pinning configs and logging seeds and model versions in your runs. The pipeline logs runtime config and parameters alongside outputs.

## License

- Repository license: Open Data Commons Attribution (ODC-By) v1.0. See `LICENSE` and the official terms at https://opendatacommons.org/licenses/by/1-0/
- Third-party datasets and materials (e.g., MIMIC, CheXpert, PubMed/S2 content) remain under their original licenses and terms. You are responsible for complying with those licenses and any applicable data use agreements.

## Citation

If you reference this research or use the released datasets, please cite:

```bibtex
@inproceedings{jang-etal-2025-medtutor,
  title     = {MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education},
  author    = {Jang, Dongsuk and Shangguan, Ziyao and Tegtmeyer, Kyle and Gupta, Anurag and Czerminski, Jan T and Chheang, Sophie and Cohan, Arman},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations},
  year      = {2025},
  url       = {https://aclanthology.org/2025.emnlp-demos.24/}
}
```

## Contact

For questions or issues, please open a GitHub issue or email to `jamesjang26@snu.ac.kr`
