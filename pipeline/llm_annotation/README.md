# LLM-as-a-Judge Annotation

This directory contains the tools for automatically evaluating the output of the main MedTutor pipeline using another powerful Large Language Model (LLM) as a "judge." This approach allows for scalable, consistent, and reproducible evaluation of generated content.

## Overview

The LLM-as-a-Judge system takes a JSON output file from the main pipeline and assesses various components (e.g., keyword appropriateness, paper relevance, feedback quality) based on predefined criteria. It generates a numerical score for each component, providing quantitative data for model comparison and analysis.

## Key Features

-   **Multi-Provider Support:** Works with different "judge" LLMs, including:
    -   OpenAI (e.g., GPT-4 series)
    -   Google Gemini (e.g., Gemini 2.5 Pro)
    -   Local HuggingFace models via `vLLM` (e.g., MedGemma)
-   **Asynchronous Execution:** Utilizes `asyncio` for high-throughput evaluation when using API-based judges like Gemini, managing rate limits effectively.
-   **Detailed Statistical Analysis:** Automatically calculates and appends a statistics summary to the output file, including mean, standard deviation, score distribution, and correlation metrics.
-   **Configuration-Driven:** All prompts, scoring criteria, and model settings are controlled via a single JSON configuration file.

## Project Structure

```
llm_annotation/
├── configs/
│     └── configs.json # Configuration for the annotator
├── run_annotation.py # Main executable script
├── annotator.py # Classes for different LLM providers
└── README.md
```


## Configuration

All settings are managed in `llm_annotation/configs/configs.json`.

-   `annotator_choice`: Select the judge model provider: `"openai"`, `"gemini"`, or `"huggingface"`.
-   `annotation_prompts`: Define the system role and the specific prompt templates and scoring criteria for each evaluation task (e.g., `keyword_appropriateness_evaluation`, `final_feedback_synthesis_evaluation`).
-   `*_annotator_settings`: Configure the specific model name and parameters for each provider. For `huggingface`, you can specify `vllm_params` like `tensor_parallel_size`.

## Usage

The evaluation is run from the command line using `run_annotation.py`.

**Basic Command:**
```bash
python llm_annotation/run_annotation.py \
    --input_file /path/to/your/pipeline_output.json \
    --output_dir /path/to/save/annotated_results \
    --config_file llm_annotation/configs/configs.json
```

### Arguments:
* --input_file: (Required) Path to the JSON output file from the main RadTutor pipeline.
* --output_dir: (Required) Directory where the annotated JSON file will be saved.
* --config_file: Path to the annotation configuration file.
* --limit: (Optional) Process only the first N cases from the input file.
* --workers: (Optional) Number of concurrent workers for async API calls (default: 3).


## Output Format
The script generates a new JSON file in the specified output directory. This file contains all the original data, with a new annotation block added to each processed report. It also includes a comprehensive statistics_summary at the top level.

```json
{
  "statistics_summary": {
    "overall_summary": { ... },
    "keyword_appropriateness_stats": { 
        "mean": 4.5, 
        "std_dev": 0.5, 
        ... 
        },
    ...
  },
  "all_processed_reports": [
    {
      "case_id": "...",
      "original_reviewer_report": "...",
      ...,
      "annotation": {
        "keyword_appropriateness": { 
            "keyword1": 5, 
            "keyword2": 4 
            },
        "paper_relevance": { "http://...": 5 },
        "textbook_summary_quality": { "keyword1": 5 },
        "mcq_quality": { 
            "MCQ_1": 5, 
            "MCQ_2": 4 
            },
        "final_feedback_quality": 5,
        "overall_quality": 5
      }
    }
  ]
}
```